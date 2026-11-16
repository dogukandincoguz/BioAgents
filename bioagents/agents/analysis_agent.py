"""Analysis Agent - Tool-executing implementation."""

import logging
from typing import Any

from langchain_core.messages import AIMessage

from bioagents.agents.helpers import create_retry_response, prepare_messages_for_agent
from bioagents.llms.llm_provider import get_llm
from bioagents.prompts.prompt_loader import load_prompt
from bioagents.prompts.tool_context import inject_prompt_context
from bioagents.tools.selected_runtime import bind_agent_node, bind_agent_tools, tools_for_agent

logger = logging.getLogger(__name__)

ANALYSIS_AGENT_PROMPT = load_prompt("analysis")

ANALYSIS_AGENT_SYSTEM_PROMPT = """
You are the Analysis Agent. Your task is to analyze biological sequences and properties.

INSTRUCTIONS:
1. Use your available tools to calculate molecular weight, isoelectric point, etc.
2. Analyze amino acid composition and protein properties
3. Identify key patterns and features
4. Provide a comprehensive analysis summary

FINAL OUTPUT FORMAT:
After analysis, respond with ONLY valid JSON (no markdown, no explanations):

{
    "sequence_properties": {
        "length": 0,
        "molecular_weight": 0.0,
        "isoelectric_point": 0.0,
        "composition": {}
    },
    "key_findings": ["finding1", "finding2"],
    "patterns_detected": ["pattern1"],
    "quality_score": "high" | "medium" | "low",
    "analysis_complete": true,
    "status": "success" | "error",
    "error": null
}

ALWAYS end with valid JSON output.
"""

ANALYSIS_AGENT_MEMORY_PROMPT = ANALYSIS_AGENT_SYSTEM_PROMPT


def create_analysis_agent(tools: list | None = None):
    """
    Create the Analysis Agent with tool-executing loop.

    Args:
        tools: List of tool objects

    Returns:
        Agent node function
    """
    llm = get_llm(prompt_name="analysis")
    tools = tools_for_agent("analysis", tools)
    llm_with_tools = bind_agent_tools(llm, tools)
    tool_names = [getattr(t, "name", str(t)) for t in tools]

    def analysis_node(state: dict[str, Any]) -> dict[str, Any]:
        """Analysis agent node function."""
        try:
            messages = state.get("messages", [])
            system_prompt = inject_prompt_context(
                ANALYSIS_AGENT_PROMPT, state=state, agent_name="analysis"
            )
            messages_to_send = prepare_messages_for_agent(
                messages, agent_name="analysis", system_prompt=system_prompt
            )

            return create_retry_response("Analysis", messages_to_send, tool_names, llm_with_tools)
        except Exception as e:
            logger.error(f"Analysis agent error: {e}", exc_info=True)
            return {
                "error": str(e),
                "messages": [AIMessage(content=f"[AGENT_ERROR] Analysis failed: {e}")],
            }

    return bind_agent_node(analysis_node)
