"""Render introductions from the active catalogue and remove disabled references."""

from __future__ import annotations

import re

from langchain_core.messages import SystemMessage

from bioagents.tools.tool_catalog import custom_entries, local_index, sdk_entries
from bioagents.tools.tool_selection import ToolSelection, get_selection

_START = "[ACTIVE TOOL CATALOGUE]"
_END = "[/ACTIVE TOOL CATALOGUE]"
_WORD = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

_NO_TOOL_SYNTHESIS_AGENTS = {
    "summary",
    "report",
    "critic",
    "notetaker",
    "note_taker",
    "user_input",
}


def get_tool_mode() -> str:
    return get_selection().mode


def redact_tool_references(text: str, selection: ToolSelection | None = None) -> str:
    selection = selection or get_selection()
    if selection.mode == "all":
        return text
    known = set(local_index()) | set(custom_entries()) | set(sdk_entries())
    aliases = {backend: alias for alias, backend in selection.aliases}

    def replace(match):
        name = match.group()
        if name in aliases:
            return aliases[name]
        return "[unavailable tool]" if name in known and name not in selection.names else name

    return _WORD.sub(replace, text)


def render_tool_catalogue(selection: ToolSelection | None = None) -> str:
    from bioagents.tools.selected_runtime import selected_metadata

    selection = selection or get_selection()
    if selection.mode == "all":
        return ""
    lines = [
        _START,
        "Only the following registered tools are enabled for this run.",
        "Use only your bound schemas. A role title does not grant additional tools.",
        "Do not import the raw tool SDK or recreate a disabled service with direct HTTP.",
        "If a tool is unavailable, check this introduction list; do not invent a replacement or result.",
    ]
    if not selection.names:
        lines.append("No registered service tools are enabled.")
    for metadata in selected_metadata(selection):
        lines.append(
            f"- {metadata['name']}: {redact_tool_references(metadata['description'], selection)}"
        )
    lines.append(_END)
    return "\n".join(lines)


def apply_tool_context(text: str, *, include_catalogue: bool = True) -> str:
    if get_selection().mode == "all":
        return text
    base = re.sub(re.escape(_START) + ".*?" + re.escape(_END), "", text, flags=re.DOTALL)

    def selected_example(match):
        example = match.group()
        return (
            "[Tool example omitted: its registered capability is unavailable.]"
            if redact_tool_references(example) != example
            else example
        )

    base = re.sub(r"```[^\n]*\n.*?```", selected_example, base, flags=re.DOTALL)
    base = redact_tool_references(base).strip()
    return base + ("\n\n" + render_tool_catalogue() if include_catalogue else "")


def build_prompt_tool_context(agent_name: str | None = None) -> str:
    """Build specialist-scoped active tool catalogue for prompt injection."""
    if agent_name:
        normalized = agent_name.strip().lower()
        if normalized in _NO_TOOL_SYNTHESIS_AGENTS:
            return ""

    if get_tool_mode() == "selected" and agent_name:
        normalized = agent_name.strip().lower()
        if normalized not in {"supervisor", "coder", "ml", "dl"}:
            from bioagents.tools.selected_runtime import ROLES, selected_metadata, tools_for_agent

            if normalized in ROLES:
                bound = tools_for_agent(normalized)
                if not bound:
                    return (
                        f"{_START}\n"
                        f"You have 0 bound tools enabled in this session. You must NOT emit <tool_call> blocks or call any tools. "
                        f"Rely on existing conversation and workspace context, or complete your analysis and instruct the "
                        f"Supervisor to route to the appropriate specialist or code agent.\n"
                        f"{_END}"
                    )

                selection = get_selection()
                bound_names = {t.name for t in bound}
                lines = [
                    _START,
                    f"Your specialist role ({agent_name}) has {len(bound)} bound tool(s) enabled for direct invocation:",
                ]
                for metadata in selected_metadata(selection):
                    if metadata["name"] in bound_names:
                        lines.append(
                            f"- [DIRECT] {metadata['name']}: {redact_tool_references(metadata['description'], selection)}"
                        )
                lines.append("")
                lines.append(
                    "Other enabled tools (do NOT call these directly; report completed findings and instruct the Supervisor to route to the owning specialist agent):"
                )
                for metadata in selected_metadata(selection):
                    if metadata["name"] not in bound_names:
                        roles = metadata.get("roles", ())
                        roles_str = f" [Agent: {', '.join(roles)}]" if roles else ""
                        lines.append(
                            f"- [HAND-OFF{roles_str}] {metadata['name']}: {redact_tool_references(metadata['description'], selection)}"
                        )
                lines.append(_END)
                return "\n".join(lines)

    return render_tool_catalogue()


def inject_prompt_context(
    prompt: str,
    state: dict | None = None,  # noqa: ARG001
    agent_name: str | None = None,
) -> str:
    """Inject scope-aware tool catalogue into agent system prompt."""
    context = build_prompt_tool_context(agent_name=agent_name)
    base = apply_tool_context(prompt, include_catalogue=False)
    if context:
        return f"{base}\n\n{context}"
    return base


def selected_system_messages(messages: list, agent_name: str | None = None) -> list:
    if get_selection().mode == "all":
        return messages
    no_tools = False
    if agent_name and agent_name.strip().lower() in _NO_TOOL_SYNTHESIS_AGENTS:
        no_tools = True
    result, included = [], False
    for message in messages:
        if isinstance(message, SystemMessage) and isinstance(message.content, str):
            should_include = (not included) and (not no_tools)
            catalog_text = (
                build_prompt_tool_context(agent_name=agent_name) if should_include else ""
            )
            cleaned = apply_tool_context(message.content, include_catalogue=False)
            new_content = f"{cleaned}\n\n{catalog_text}" if catalog_text else cleaned
            message = message.model_copy(update={"content": new_content})
            included = True
        result.append(message)
    if not included and not no_tools:
        result.insert(0, SystemMessage(content=build_prompt_tool_context(agent_name=agent_name)))
    return result
