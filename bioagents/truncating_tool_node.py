"""Wrap LangGraph ToolNode so tool results are capped before entering message state."""

from __future__ import annotations

import contextlib
import json
import logging
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.prebuilt import ToolNode

from bioagents.limits import MAX_TOOL_OUTPUT_CHARS
from bioagents.tools.tool_policy import ToolPolicy, get_default_policy
from bioagents.tools.tool_retry import ainvoke_with_retry, invoke_with_retry

logger = logging.getLogger(__name__)


def _text_len(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        n = 0
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                n += len(str(block.get("text", "")))
            elif isinstance(block, str):
                n += len(block)
        return n
    return len(str(content))


def _truncate_content(content: Any, max_chars: int) -> Any:
    if max_chars <= 0:
        return content
    if _text_len(content) <= max_chars:
        return content
    if isinstance(content, str):
        return content[:max_chars] + "\n... [truncated by BioAgents: output size cap]"
    if isinstance(content, list):
        out = []
        used = 0
        for block in content:
            if used >= max_chars:
                break
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text", ""))
                take = min(len(text), max_chars - used)
                nb = dict(block)
                nb["text"] = text[:take] + ("\n... [truncated]" if take < len(text) else "")
                out.append(nb)
                used += take
            else:
                s = str(block)
                take = min(len(s), max_chars - used)
                out.append(s[:take])  # type: ignore[arg-type]
                used += take
        return out
    return str(content)[:max_chars] + "\n... [truncated by BioAgents: output size cap]"


def truncate_tool_messages_payload(payload: Any, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> Any:
    """Truncate ToolMessage contents inside a ToolNode return value (dict, list, or ToolMessage)."""
    if isinstance(payload, ToolMessage):
        return ToolMessage(
            content=_truncate_content(payload.content, max_chars),
            tool_call_id=payload.tool_call_id,
            name=getattr(payload, "name", "") or "",
            status=getattr(payload, "status", "success"),
        )

    if isinstance(payload, dict) and "messages" in payload:
        msgs = payload["messages"]
        if isinstance(msgs, list):
            new_msgs = [truncate_tool_messages_payload(m, max_chars) for m in msgs]
            return {**payload, "messages": new_msgs}
        return payload

    if isinstance(payload, list):
        return [truncate_tool_messages_payload(m, max_chars) for m in payload]

    return payload


def _requested_calls(input: Any) -> list:
    if isinstance(input, list):
        if input and isinstance(input[-1], dict) and input[-1].get("type") == "tool_call":
            calls = input
        else:
            messages = input
            last = messages[-1] if messages else None
            calls = last.tool_calls if isinstance(last, AIMessage) else []
    else:
        messages = (
            input.get("messages", []) if isinstance(input, dict) else getattr(input, "messages", [])
        )
        last = messages[-1] if messages else None
        calls = last.tool_calls if isinstance(last, AIMessage) else []

    if isinstance(calls, list):
        for tc in calls:
            if isinstance(tc, dict):
                raw_name = str(tc.get("name") or "").strip()
                if "<" in raw_name or ">" in raw_name or " " in raw_name or "\n" in raw_name:
                    clean_name = re.split(r"[<\s>]", raw_name, maxsplit=1)[0].strip()
                    if clean_name:
                        tc["name"] = clean_name
    return calls


def _defer_tool_batch(calls: list, blocked_index: int, denial: ToolMessage) -> dict:
    """Close all call IDs without executing a partially available batch."""
    messages = []
    for index, call in enumerate(calls):
        if index == blocked_index:
            messages.append(denial)
        else:
            messages.append(
                ToolMessage(
                    content=json.dumps(
                        {
                            "status": "error",
                            "error_code": "batch_deferred",
                            "message": "No calls in this batch executed. Check the current tool introduction list and submit only available tools.",
                            "execution_state": "not_started",
                            "retryable": False,
                        }
                    ),
                    name=call.get("name", "unknown"),
                    tool_call_id=call.get("id") or str(uuid.uuid4()),
                    status="error",
                )
            )
    return {"messages": messages}


def _selection_denial(input: Any, selection, tool_names: set[str]) -> dict | None:
    from bioagents.tools.tool_selection import unavailable_tool

    calls = _requested_calls(input)
    for index, call in enumerate(calls):
        name = call.get("name", "unknown")
        if not selection.allows(name):
            return _defer_tool_batch(
                calls,
                index,
                ToolMessage(
                    content=json.dumps(unavailable_tool(name)),
                    name=name,
                    tool_call_id=call.get("id") or str(uuid.uuid4()),
                    status="error",
                ),
            )
        elif name not in tool_names:
            from bioagents.tools.tool_catalog import get_tool_catalog

            catalog_entry = None
            with contextlib.suppress(Exception):
                catalog_entry = get_tool_catalog().get(name)
            roles = ()
            if catalog_entry:
                roles = catalog_entry.get("roles") or catalog_entry.get("agent_roles") or ()
            roles_str = f" [{', '.join(roles)}]" if roles else ""
            denial_msg = {
                "status": "wrong_agent_for_tool",
                "error_code": "wrong_agent_for_tool",
                "tool_name": name,
                "owning_agents": list(roles),
                "message": (
                    f"Tool '{name}' is not bound to this agent. It is owned by{roles_str} specialist agent(s). "
                    "Do not retry this tool yourself; complete your current scope and ask the Supervisor to route to the appropriate specialist or sandbox agent."
                ),
                "execution_state": "not_started",
                "retryable": False,
            }
            return _defer_tool_batch(
                calls,
                index,
                ToolMessage(
                    content=json.dumps(denial_msg),
                    name=name,
                    tool_call_id=call.get("id") or str(uuid.uuid4()),
                    status="error",
                ),
            )
    return None


class TruncatingToolRunnable(Runnable):
    """ToolNode wrapper that caps tool outputs before they are merged into state."""

    def __init__(self, tools: list):
        super().__init__()
        from bioagents.tools.tool_selection import get_selection

        self._selection = get_selection()
        self._tool_names = {tool.name for tool in tools}
        self._bound_tool_names = self._tool_names
        self._inner = ToolNode(tools)
        self._max_chars = MAX_TOOL_OUTPUT_CHARS

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        from bioagents.tools.tool_selection import selection_scope

        with selection_scope(self._selection):
            denial = _selection_denial(input, self._selection, self._bound_tool_names)
            if denial is not None:
                return denial
            out = invoke_with_retry(self._inner.invoke, input, config, **kwargs)
            return truncate_tool_messages_payload(out, self._max_chars)

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        from bioagents.tools.tool_selection import selection_scope

        with selection_scope(self._selection):
            denial = _selection_denial(input, self._selection, self._bound_tool_names)
            if denial is not None:
                return denial
            out = await ainvoke_with_retry(self._inner.ainvoke, input, config, **kwargs)
            return truncate_tool_messages_payload(out, self._max_chars)


def make_truncating_tool_node(tools: list) -> TruncatingToolRunnable:
    """Build a ToolNode that caps every tool result before it is merged into graph state."""
    return TruncatingToolRunnable(tools)


class ApprovalToolRunnable(Runnable):
    """ToolNode wrapper that adds policy evaluation + truncation.

    Policy check results are embedded in ToolMessage content so the frontend
    can display approval requests.  Tools requiring approval are blocked on
    first call (returning a denial message).  If the user approves the tool
    via the frontend, it is added to the session's approved list and the
    agent's next retry will succeed.
    """

    def __init__(
        self,
        tools: list,
        policy: ToolPolicy | None = None,
    ):
        super().__init__()
        from bioagents.tools.tool_selection import get_selection

        self._selection = get_selection()
        self._tool_names = {tool.name for tool in tools}
        self._bound_tool_names = self._tool_names
        self._inner = ToolNode(tools)
        self._max_chars = MAX_TOOL_OUTPUT_CHARS
        self._policy = policy or get_default_policy()

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        from bioagents.tools.tool_selection import selection_scope

        with selection_scope(self._selection):
            return self._invoke(input, config, **kwargs)

    def _invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        """Execute tools with policy check and truncation."""
        denial = _selection_denial(input, self._selection, self._bound_tool_names)
        if denial is not None:
            return denial
        for tc in _requested_calls(input):
            tool_name = tc.get("name", "unknown")
            tool_args = tc.get("args", {})
            policy_result = self._policy.evaluate(tool_name, tool_args)

            if not policy_result.allowed:
                logger.warning(
                    "Policy blocked tool '%s': %s",
                    tool_name,
                    policy_result.reason,
                )
                denial_msg = ToolMessage(
                    content=(
                        f"[POLICY_BLOCKED] Tool '{tool_name}' was blocked: "
                        f"{policy_result.reason}. Try a different approach."
                    ),
                    tool_call_id=tc.get("id", str(uuid.uuid4())),
                    name=tool_name,
                    status="error",
                )
                return {"messages": [denial_msg]}

            if policy_result.requires_approval:
                # Block with the approval info so the frontend can show the approval UI.
                request_id = str(uuid.uuid4())
                logger.info(
                    "Tool '%s' requires approval (request_id=%s): %s",
                    tool_name,
                    request_id,
                    policy_result.reason,
                )
                denial_msg = ToolMessage(
                    content=(
                        f"[APPROVAL_REQUIRED] Tool '{tool_name}' requires user approval. "
                        f"Reason: {policy_result.reason}. "
                        f"Risk level: {policy_result.risk_level}. "
                        f"If approved, retry the tool call. "
                        f"[approval_request_id={request_id}]"
                    ),
                    tool_call_id=tc.get("id", str(uuid.uuid4())),
                    name=tool_name,
                    status="error",
                )
                return {"messages": [denial_msg]}

        # Execute the tools, retrying transient failures so a blip is not the agent's answer.
        out = invoke_with_retry(self._inner.invoke, input, config, **kwargs)
        return truncate_tool_messages_payload(out, self._max_chars)

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        """Async version — delegates to sync invoke (ToolNode is sync internally)."""
        return self.invoke(input, config, **kwargs)


def make_approval_tool_node(
    tools: list,
    policy: ToolPolicy | None = None,
) -> ApprovalToolRunnable:
    """Build an ApprovalToolRunnable with policy gate + truncation."""
    return ApprovalToolRunnable(tools, policy=policy)
