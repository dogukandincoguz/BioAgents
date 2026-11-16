"""Unit tests for routing guards, summary sanitization, and tool resilience."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import requests
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from bioagents.agents.summary_agent import _is_invalid_summary_response, create_summary_agent
from bioagents.agents.supervisor_agent import _has_recent_autonomous_directive, _is_autonomous_mode
from bioagents.agents.supervisor_helpers import apply_post_llm_routing_overrides
from bioagents.graph import _detect_consecutive_duplicate_calls
from bioagents.limits import MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS
from bioagents.tools.literature_tools import _throttled_ncbi_get
from bioagents.tools.proteomics_tools import (
    _search_uniprot_fasta_fallback,
    fetch_uniprot_fasta_impl,
)
from bioagents.tools.tool_catalog import build_tool_catalog
from bioagents.truncating_tool_node import _selection_denial


def _make_tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    msg = AIMessage(content="", name="Genomics")
    msg.tool_calls = [{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    return msg


def test_duplicate_call_detector_exempts_transient_errors() -> None:
    """Genuine transient errors should not count toward duplicate loop breaking."""
    args = {"sequence": "ACDEFGHIKL"}
    messages: list = [HumanMessage(content="Run blast")]
    for i in range(MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS + 1):
        messages.append(_make_tool_call("run_blast_search", args, f"c{i}"))
        messages.append(
            ToolMessage(
                content="HTTP status 429: Too Many Requests",
                tool_call_id=f"c{i}",
                name="run_blast_search",
                status="error",
            )
        )

    is_loop, _ = _detect_consecutive_duplicate_calls(messages, "Genomics")
    assert is_loop is False


def test_duplicate_call_detector_does_not_treat_numeric_ids_as_transient() -> None:
    """Normal outputs containing '429' or '503' digits (e.g. PMID 39503123) must still trip loops."""
    args = {"protein_id": "P24941"}
    messages: list = [HumanMessage(content="Fetch protein")]
    for i in range(MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS):
        messages.append(_make_tool_call("fetch_uniprot_fasta", args, f"c{i}"))
        messages.append(
            ToolMessage(
                content='{"pmid": "39503123", "length": 429}',
                tool_call_id=f"c{i}",
                name="fetch_uniprot_fasta",
            )
        )

    is_loop, desc = _detect_consecutive_duplicate_calls(messages, "Genomics")
    assert is_loop is True
    assert "fetch_uniprot_fasta" in desc


def test_invalid_summary_response_detects_tool_calls_and_intents() -> None:
    """Raw XML tool calls, structured tool_calls, and intent-only responses are rejected."""
    history = [
        HumanMessage(content="Run workflow"),
        AIMessage(content="Detailed scientific findings with AUROC 0.98 and artifacts.", name="ML"),
    ]
    raw_xml = AIMessage(content="<tool_call>fetch_uniprot_fasta</tool_call>")
    intent = AIMessage(content="I'll run the BLAST query now.")
    structured = AIMessage(
        content="Calling tool",
        tool_calls=[{"name": "run_blast_search", "args": {}, "id": "t1", "type": "tool_call"}],
    )

    assert _is_invalid_summary_response(raw_xml, "full_summary", history) is True
    assert _is_invalid_summary_response(intent, "full_summary", history) is True
    assert _is_invalid_summary_response(structured, "full_summary", history) is True


@patch("bioagents.agents.summary_agent.get_llm")
@patch("bioagents.agents.summary_agent.load_prompt", return_value="Summary prompt")
def test_summary_fallback_returns_clean_aimessage_without_tool_calls(
    _mock_load: Mock, mock_get_llm: Mock
) -> None:
    """Fallback must return a fresh AIMessage without leaking tool_calls metadata."""
    bad_response = AIMessage(
        content="<tool_call>foo</tool_call>",
        tool_calls=[{"name": "foo", "args": {}, "id": "id1", "type": "tool_call"}],
    )
    mock_llm = Mock()
    mock_llm.invoke = Mock(return_value=bad_response)
    mock_get_llm.return_value = mock_llm

    node = create_summary_agent()
    result = node(
        {
            "messages": [
                HumanMessage(content="Analyze CDK2"),
                AIMessage(content="Docking affinity was -6.706 kcal/mol.", name="Docking"),
            ]
        }
    )

    out_msg = result["messages"][0]
    assert isinstance(out_msg, AIMessage)
    assert not getattr(out_msg, "tool_calls", None)
    assert "-6.706 kcal/mol" in out_msg.content


@patch("bioagents.tools.proteomics_tools.requests.get")
def test_uniprot_fallback_handles_request_exceptions_and_recovers(mock_get: Mock) -> None:
    """Fallback should recover FASTA on 404 and swallow RequestException cleanly."""
    not_found = Mock(status_code=404, text="")
    search_hit = Mock(status_code=200, text=">sp|P20638|F13L\nMWPFASVP")
    mock_get.side_effect = [not_found, search_hit]

    res = fetch_uniprot_fasta_impl("F13L_VACCC")
    assert res.startswith(">sp|P20638|F13L")

    mock_get.side_effect = requests.Timeout("timed out")
    assert _search_uniprot_fasta_fallback("F13L_VACCC") is None


@patch("bioagents.tools.literature_tools.time.sleep")
@patch("bioagents.tools.literature_tools.requests.get")
def test_throttled_ncbi_get_retries_transient_status(mock_get: Mock, _mock_sleep: Mock) -> None:
    """_throttled_ncbi_get retries on HTTP 429/503 and succeeds when subsequent call returns 200."""
    rate_limited = Mock(status_code=429)
    ok_resp = Mock(status_code=200)
    mock_get.side_effect = [rate_limited, ok_resp]

    resp = _throttled_ncbi_get("https://eutils.ncbi.nlm.nih.gov/test", {"db": "pubmed"})
    assert resp is ok_resp
    assert mock_get.call_count == 2

    mock_get.side_effect = requests.ConnectionError("down")
    with pytest.raises(requests.ConnectionError):
        _throttled_ncbi_get("https://eutils.ncbi.nlm.nih.gov/test", {"db": "pubmed"})


def test_autonomous_mode_and_directive_dedup() -> None:
    """Autonomous mode triggers only from HumanMessage directives and detects recent duplicates."""
    assert _is_autonomous_mode([AIMessage(content="Run autonomously")]) is False
    assert _is_autonomous_mode([HumanMessage(content="Do not ask for user input.")]) is True

    directive = HumanMessage(content="[AUTONOMOUS MODE DIRECTIVE] proceed", name="System")
    assert _has_recent_autonomous_directive([directive]) is True
    assert _has_recent_autonomous_directive([HumanMessage(content="normal")]) is False


def test_apply_post_llm_routing_overrides_skips_tool_builder_without_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supervisor must not override to tool_builder if tool_builder has 0 bound tools."""
    messages = [
        HumanMessage(content="Train model"),
        AIMessage(content="missing tool for this step", name="ML"),
    ]
    monkeypatch.setattr("bioagents.tools.selected_runtime.tools_for_agent", lambda _role: [])

    agent, _ = apply_post_llm_routing_overrides("coder", messages, ["coder", "tool_builder"])
    assert agent == "coder"


def test_tool_catalog_enforces_strict_specialist_roles() -> None:
    """build_tool_catalog maps docking and blast tools strictly to their specialist roles."""
    catalog = build_tool_catalog()
    assert catalog["run_docking"]["roles"] == ("docking",)
    assert catalog["run_blast_search"]["roles"] == ("genomics", "phylogenetics")
    assert catalog["search_pubmed"]["roles"] == ("literature", "research")


def test_selection_denial_reports_wrong_agent_for_tool() -> None:
    """Calling an enabled tool not bound to the current agent yields wrong_agent_for_tool."""
    from bioagents.tools.tool_selection import ToolSelection

    selection = ToolSelection(mode="selected", local_tools=("run_docking",))
    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "run_docking", "args": {}, "id": "call_1", "type": "tool_call"}],
    )
    denial = _selection_denial({"messages": [ai_msg]}, selection, {"search_pubmed"})
    assert denial is not None
    tool_msg = denial["messages"][0]
    assert "wrong_agent_for_tool" in tool_msg.content
    assert "docking" in tool_msg.content
