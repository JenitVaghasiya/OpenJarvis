"""Tests for the faithful ToolOrchestra unified-tool registry."""

from __future__ import annotations

import pytest

from openjarvis.agents.hybrid.expert_registry import (
    CATEGORY_BASIC,
    CATEGORY_CLOUD_FRONTIER,
    CATEGORY_LOCAL_OSS,
    KIND_MODEL,
    KIND_TOOL,
    ExpertTool,
    build_tool_specs,
    openjarvis_tool,
    orchestrator_catalog,
    to_worker_dict,
    tools_by_name,
)


def test_catalog_names_unique_and_valid():
    cat = orchestrator_catalog()
    names = [t.name for t in cat]
    assert len(names) == len(set(names))
    assert all(isinstance(t, ExpertTool) for t in cat)


def test_invalid_tool_rejected():
    with pytest.raises(ValueError):
        ExpertTool(name="x", kind="bogus", backend_type="openai", summary="", model="m")
    with pytest.raises(ValueError):
        ExpertTool(
            name="x", kind=KIND_MODEL, backend_type="openai", summary="", model=None
        )


def test_specs_shape_and_pricing_in_description():
    cat = orchestrator_catalog()
    specs = build_tool_specs(cat)
    by = {s["function"]["name"]: s for s in specs}
    gpt = by["gpt_5_5"]
    assert gpt["type"] == "function"
    assert "input" in gpt["function"]["parameters"]["properties"]
    # Price table is surfaced in the description (the policy is trained on it).
    assert "/1M input" in gpt["function"]["description"]
    # Search tool takes a query, code takes code.
    assert "query" in by["web_search"]["function"]["parameters"]["properties"]
    assert "code" in by["code_interpreter"]["function"]["parameters"]["properties"]


# Two cloud-frontier + four local-OSS model tools, in catalog order.
_ORCH_MODEL_NAMES = [
    "gpt_5_5",
    "claude_opus_4_8",
    "qwen3_5_9b",
    "qwen3_6_27b_fp8",
    "qwen3_5_122b_a10b_fp8",
    "qwen3_5_397b_a17b_fp8",
]

# Bridged real OpenJarvis tools (basic) appended after web_search/code_interpreter.
_ORCH_BASIC_NAMES = [
    "web_search",
    "code_interpreter",
    "calculator",
    "shell_exec",
    "file_read",
    "file_write",
    "http_request",
    "think",
    "apply_patch",
    "pdf_extract",
    "db_query",
]


def test_orchestrator_catalog_two_model_classes_plus_basics():
    cat = orchestrator_catalog()
    names = [t.name for t in cat]
    # 6 model tools (2 cloud_frontier + 4 local_oss) come first.
    assert names[:6] == _ORCH_MODEL_NAMES
    # then the basic tools.
    assert set(_ORCH_BASIC_NAMES) <= set(names)
    assert len(cat) == 6 + len(_ORCH_BASIC_NAMES)
    by = tools_by_name(cat)
    assert by["gpt_5_5"].category == CATEGORY_CLOUD_FRONTIER
    assert by["claude_opus_4_8"].category == CATEGORY_CLOUD_FRONTIER
    # Default routing for every model tool is OpenRouter (no servers required).
    for n in (
        "qwen3_5_9b",
        "qwen3_6_27b_fp8",
        "qwen3_5_122b_a10b_fp8",
        "qwen3_5_397b_a17b_fp8",
    ):
        assert by[n].category == CATEGORY_LOCAL_OSS
        assert by[n].backend_type == "openrouter"
        assert by[n].base_url is None
        # OpenRouter routing carries the slug + a (estimated) per-token price.
        assert "/" in by[n].model and by[n].price_in > 0.0


def test_orchestrator_catalog_categories_present():
    cat = orchestrator_catalog()
    cats = {t.category for t in cat}
    assert cats == {CATEGORY_CLOUD_FRONTIER, CATEGORY_LOCAL_OSS, CATEGORY_BASIC}


def test_orchestrator_can_drop_tools():
    cat = orchestrator_catalog(include_tools=False)
    assert {t.category for t in cat} == {CATEGORY_CLOUD_FRONTIER, CATEGORY_LOCAL_OSS}
    assert len(cat) == 6


def test_orchestrator_specs_include_category_field():
    specs = build_tool_specs(orchestrator_catalog())
    by = {s["function"]["name"]: s for s in specs}
    assert by["gpt_5_5"]["function"]["category"] == CATEGORY_CLOUD_FRONTIER
    assert by["qwen3_5_9b"]["function"]["category"] == CATEGORY_LOCAL_OSS
    assert by["web_search"]["function"]["category"] == CATEGORY_BASIC
    assert by["shell_exec"]["function"]["category"] == CATEGORY_BASIC
    # Every tool carries a category tag.
    assert all("category" in s["function"] for s in specs)


def test_orchestrator_local_models_get_base_url_when_provided():
    # An endpoint switches that model from the OpenRouter default to local vLLM.
    cat = orchestrator_catalog(
        local_endpoints={
            "Qwen/Qwen3.5-9B": "http://x/v1",
            "Qwen/Qwen3.6-27B-FP8": "http://y/v1",
        }
    )
    by = tools_by_name(cat)
    assert by["qwen3_5_9b"].backend_type == "vllm"
    assert by["qwen3_5_9b"].base_url == "http://x/v1"
    assert by["qwen3_5_9b"].price_in == 0.0 and by["qwen3_5_9b"].price_out == 0.0
    assert by["qwen3_6_27b_fp8"].base_url == "http://y/v1"
    # Unmapped local model -> stays on the OpenRouter default (base_url None).
    assert by["qwen3_5_122b_a10b_fp8"].backend_type == "openrouter"
    assert by["qwen3_5_122b_a10b_fp8"].base_url is None
    # Cloud frontier carries real pricing.
    assert by["claude_opus_4_8"].price_in > 0.0
    assert by["gpt_5_5"].price_in > 0.0


def test_orchestrator_model_backends_override():
    # Force a cloud model onto OpenRouter and a local model onto vLLM explicitly.
    cat = orchestrator_catalog(
        model_backends={
            "claude-opus-4-8": "openrouter",
            "Qwen/Qwen3.5-397B-A17B-FP8": "vllm",
        },
        local_endpoints={"Qwen/Qwen3.5-397B-A17B-FP8": "http://z/v1"},
    )
    by = tools_by_name(cat)
    assert by["claude_opus_4_8"].backend_type == "openrouter"
    assert by["claude_opus_4_8"].model == "anthropic/claude-opus-4.8"
    # No override for gpt-5.5 -> it defaults to its NATIVE first-party API
    # (openai), not OpenRouter. Frontier models hit their native provider by
    # default; OpenRouter is only the fallback for OSS/local models or when
    # explicitly requested via model_backends.
    assert by["gpt_5_5"].backend_type == "openai"  # native default
    assert by["gpt_5_5"].model == "gpt-5.5"
    assert by["qwen3_5_397b_a17b_fp8"].backend_type == "vllm"
    assert by["qwen3_5_397b_a17b_fp8"].base_url == "http://z/v1"


def test_orchestrator_openrouter_slug_override():
    cat = orchestrator_catalog(
        openrouter_slugs={"Qwen/Qwen3.5-9B": "qwen/qwen3.5-9b-custom"}
    )
    by = tools_by_name(cat)
    assert by["qwen3_5_9b"].model == "qwen/qwen3.5-9b-custom"


def test_openjarvis_tool_bridges_real_tool_with_custom_schema():
    params = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }
    t = openjarvis_tool("shell_exec", summary="Run a shell command.", params=params)
    assert t.kind == KIND_TOOL
    assert t.backend_type == "openjarvis-tool"
    assert t.model == "shell_exec"
    assert t.category == CATEGORY_BASIC
    spec = t.to_spec()
    assert spec["function"]["name"] == "shell_exec"
    assert spec["function"]["parameters"] == params
    assert spec["function"]["category"] == CATEGORY_BASIC


def test_build_tool_specs_includes_category_for_bridged_tools():
    specs = build_tool_specs(
        [
            openjarvis_tool(
                "calculator",
                summary="Math.",
                params={
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            ),
        ]
    )
    assert specs[0]["function"]["category"] == CATEGORY_BASIC
    assert "expression" in specs[0]["function"]["parameters"]["properties"]


def test_to_worker_dict_maps_backend():
    cat = orchestrator_catalog(local_endpoints={"Qwen/Qwen3.5-9B": "http://x/v1"})
    by = tools_by_name(cat)
    assert to_worker_dict(by["gpt_5_5"]) == {
        "name": "gpt_5_5",
        "type": "openai",
        "model": "gpt-5.5",
    }
    local = to_worker_dict(by["qwen3_5_9b"])
    assert local["type"] == "vllm" and local["base_url"] == "http://x/v1"
