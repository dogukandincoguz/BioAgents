"""Summary Agent - Final user-facing output layer."""

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from bioagents.agents.helpers import (
    extract_best_content,
    invoke_with_retry,
    is_empty_response,
    prepare_messages_for_agent,
)
from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt
from bioagents.tools.selected_runtime import bind_agent_node

logger = logging.getLogger(__name__)

SUMMARY_AGENT_PROMPT = load_prompt("summary")


def analyze_execution_mode(messages: list) -> tuple[str, str]:
    """
    Analyze the message history to determine if this was a simple single-turn query
    or a complex multi-agent execution.

    Returns:
        Tuple of (mode, reasoning)
    """
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]

    total_tool_calls = 0
    for msg in ai_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            total_tool_calls += len(msg.tool_calls)

    is_simple_query = len(human_messages) == 1 and total_tool_calls == 0 and len(ai_messages) <= 1

    if is_simple_query:
        return (
            "direct_answer",
            "Single user query with no tool calls or agent routing detected.",
        )
    else:
        return (
            "full_summary",
            f"Complex execution: {len(human_messages)} user messages, "
            f"{total_tool_calls} tool calls, {len(ai_messages)} agent messages.",
        )


def _is_invalid_summary_response(response: Any, summary_mode: str, messages: list) -> bool:
    """Check if the summary agent returned an empty, raw-tool-call, or hallucinated response."""
    if is_empty_response(response):
        return True
    if getattr(response, "tool_calls", None):
        return True
    text = str(getattr(response, "content", "") or "").strip()
    if "<tool_call>" in text or "</tool_call>" in text or "<arg_key>" in text:
        return True
    if re.match(
        r"^(I'll\s+run|I\s+will\s+run|Let\s+me\s+run|I\s+will\s+now\s+call|I'll\s+now\s+execute)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    msg_list = messages or []
    return bool(
        summary_mode == "full_summary"
        and len(text) < 120
        and len(msg_list) > 4
        and len(extract_best_content(msg_list)) > len(text)
    )


def create_summary_agent():
    """
    Create the Summary Agent.

    Returns:
        Agent node function
    """
    load_prompt("summary")
    llm = get_llm(prompt_name="summary")

    def summary_node(state):
        """Summary agent - generates final user output."""
        messages = state.get("messages", []) or []

        mode, _reasoning = analyze_execution_mode(messages)

        mode_instruction = (
            "EXECUTION MODE: DIRECT ANSWER\n"
            "The user asked a simple question with no agent involvement.\n"
            "Provide a direct, concise answer without mentioning agents or tools.\n"
            "Keep the response focused and natural."
            if mode == "direct_answer"
            else "EXECUTION MODE: FULL SUMMARY\n"
            "Multiple agents and tools were involved in this execution.\n"
            "Summarize the workflow, agents used, tools called, and key results.\n"
            "Use clear section headers and maintain a user-friendly tone."
        )

        windowed = prepare_messages_for_agent(messages, "summary", summary_mode=True)
        messages_with_system = [
            SystemMessage(content=SUMMARY_AGENT_PROMPT),
            SystemMessage(content=mode_instruction),
            *windowed,
        ]

        result = invoke_with_retry("Summary", llm, messages_with_system, max_retries=1)

        response = result["messages"][0]
        if _is_invalid_summary_response(response, mode, messages):
            logger.warning(
                "Invalid summary response (%r...); falling back to extract_best_content",
                str(getattr(response, "content", ""))[:80],
            )
            best = extract_best_content(messages)
            fallback_content = (
                best
                if best
                else (
                    "The workflow completed but was unable to generate a summary. "
                    "Please review the agent outputs in the audit log for details."
                )
            )
            return {"messages": [AIMessage(content=fallback_content, name="Summary")]}

        return result

    return bind_agent_node(summary_node)


def format_memory_for_summary(memory: dict) -> str:
    """Format memory for summary generation."""
    lines = []

    for agent_name in sorted(memory.keys()):
        agent_data = memory[agent_name]

        if agent_data.get("status") != "success":
            continue

        lines.append(f"\n{agent_name.upper()}:")

        if agent_data.get("data"):
            try:
                lines.append(json.dumps(agent_data["data"], indent=2))
            except (TypeError, ValueError):
                lines.append(str(agent_data["data"]))

    return "\n".join(lines) if lines else "No completed agent tasks."
