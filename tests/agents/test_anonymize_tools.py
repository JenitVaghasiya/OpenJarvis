"""Unit coverage for ``anonymize_tools`` — the identity-stripping step that makes
routing data unbiased. See ``expert_registry.anonymize_tools``.

The anonymizer takes the orchestrator catalog and replaces every MODEL expert
with an opaque ``model_xxxx`` label, a uniform brand-free description, no
price/latency line, and shuffles the expert block — so the policy can't route on
a model's name, position, cost, or tier. Basic tools keep their real names.
It returns ``(anon_tools, anon_to_real)``.
"""

from __future__ import annotations

import json
import random

from openjarvis.agents.hybrid.expert_registry import (
    KIND_MODEL,
    anonymize_tools,
    build_tool_specs,
    orchestrator_catalog,
    tools_by_name,
)

# Real brand tokens that must never leak into the anonymized, model-facing specs.
_BRANDS = ("gpt", "claude", "gemini", "qwen")


def _model_names(cat):
    return [t.name for t in cat if t.kind == KIND_MODEL]


def _basic_names(cat):
    return [t.name for t in cat if t.kind != KIND_MODEL]


def test_no_brand_names_in_anonymized_specs():
    cat = orchestrator_catalog()
    anon, _ = anonymize_tools(cat, random.Random(0))
    specs = build_tool_specs(anon)
    # The whole model-facing payload (names + descriptions + categories) must be
    # brand-free. `.model` is preserved on the tool for dispatch but is NOT part
    # of the spec the orchestrator conditions on, so it's fine that it still holds
    # the real id.
    blob = json.dumps(specs).lower()
    for brand in _BRANDS:
        assert brand not in blob, f"brand {brand!r} leaked into anonymized specs"
    # And specifically the anonymized expert descriptions carry no brand.
    for t in anon:
        if t.name.startswith("model_"):
            assert not any(b in t.description().lower() for b in _BRANDS)


def test_hide_cost_removes_price_line_from_descriptions():
    cat = orchestrator_catalog()
    # Sanity: the raw model tools DO surface a price line before anonymizing.
    raw_model = next(t for t in cat if t.kind == KIND_MODEL)
    assert "Pricing:" in raw_model.description()

    anon, anon_to_real = anonymize_tools(cat, random.Random(1))
    for t in anon:
        if t.name in anon_to_real:  # an anonymized model expert
            assert t.hide_cost is True
            desc = t.description()
            assert "Pricing:" not in desc
            assert "$" not in desc
            assert "/1M" not in desc
            assert "latency" not in desc.lower()


def test_each_model_maps_to_opaque_label_and_round_trips():
    cat = orchestrator_catalog()
    orig_models = _model_names(cat)
    anon, anon_to_real = anonymize_tools(cat, random.Random(2))

    # One opaque label per real model, all in the model_xxxx namespace.
    assert len(anon_to_real) == len(orig_models)
    assert all(lbl.startswith("model_") for lbl in anon_to_real)

    # No collisions: labels unique, and each real model recovered exactly once.
    assert len(set(anon_to_real)) == len(anon_to_real)
    assert len(set(anon_to_real.values())) == len(anon_to_real)
    assert set(anon_to_real.values()) == set(orig_models)

    # Round-trip: every anonymized expert's label resolves back to a real name,
    # and `.model` is preserved so dispatch still reaches the right backend.
    by_orig = tools_by_name(cat)
    for t in anon:
        if t.name.startswith("model_"):
            real = anon_to_real[t.name]
            assert real in by_orig
            assert t.model == by_orig[real].model  # backend id untouched

    # Basic tools keep their real names and are untouched by the label map.
    basics = _basic_names(cat)
    anon_names = {t.name for t in anon}
    assert set(basics) <= anon_names
    assert not (set(basics) & set(anon_to_real))


def test_experts_block_on_top_then_basics():
    cat = orchestrator_catalog()
    anon, anon_to_real = anonymize_tools(cat, random.Random(3))
    n_models = len(anon_to_real)
    # Experts form a contiguous block at the front, basics underneath.
    assert all(t.name.startswith("model_") for t in anon[:n_models])
    assert all(not t.name.startswith("model_") for t in anon[n_models:])


def test_labels_and_order_are_shuffled_not_identity():
    cat = orchestrator_catalog()
    orig_models = _model_names(cat)

    orderings = set()
    labels_for_first = set()
    for seed in range(25):
        anon, anon_to_real = anonymize_tools(cat, random.Random(seed))
        # Real-model order as it appears in the anonymized expert block.
        order = tuple(anon_to_real[t.name] for t in anon if t.name.startswith("model_"))
        orderings.add(order)
        # Label assigned to the first catalog model varies across rngs.
        rev = {real: lbl for lbl, real in anon_to_real.items()}
        labels_for_first.add(rev[orig_models[0]])

    # Not a fixed identity: across seeds the expert order actually varies...
    assert len(orderings) > 1
    # ...and at least one ordering differs from the input model order.
    assert any(order != tuple(orig_models) for order in orderings)
    # Labels are random per-rng, not a stable function of position.
    assert len(labels_for_first) > 1
