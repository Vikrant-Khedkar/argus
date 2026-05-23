"""Backward-compatibility smoke: every v1 import surface still resolves
through the root-level shims and points at the v2 implementations under
``argus.*``. This is the contract that lets the existing 70-row eval,
``cli.py``, and ``app.py`` keep working without edits.

Run with: ``uv run python -m pytest tests/test_backward_compat.py -q``
"""

from __future__ import annotations

import importlib


def test_adapter_shim_re_exports_argus_classes():
    import adapter
    import argus

    assert adapter.ChatProvider is argus.ChatProvider
    assert adapter.ModalProvider is argus.ModalProvider
    assert adapter.OpenRouterProvider is argus.OpenRouterProvider
    assert callable(adapter.get_provider)


def test_v1_module_imports():
    for name in ("assistant", "risk_score", "memory", "observability", "cli"):
        importlib.import_module(name)


def test_v1_risk_scorer_still_works_and_has_v2_extension():
    from risk_score import RiskScorer, AXES, RUBRICS

    assert AXES == (
        "output_liability",
        "discrimination_liability",
        "safety_liability",
    )
    for axis in AXES:
        assert axis in RUBRICS

    rs = RiskScorer()
    assert rs.judge_model == "anthropic/claude-sonnet-4"
    # v2 extension added in Phase 8
    assert hasattr(rs, "score_conversation")


def test_full_v2_surface_resolves():
    expected = {
        # core
        "Evaluator", "EvalConfig", "AuditWriter", "AuditIndex",
        "AuditReport", "AuditRow",
        # multi-turn
        "score_conversation", "ConversationProbe", "ConversationTurn",
        "DEFAULT_MULTI_TURN_PROBES",
        # providers
        "ChatProvider", "ModalProvider", "OpenRouterProvider",
        "HTTPProvider", "HuggingFaceProvider", "GuardrailedProvider",
        # guardrails
        "PreFlightPatternGuard", "PostFlightRegenGuard",
        "PostFlightHardRefuseGuard",
        # scorers
        "Scorer", "ModelBasedScorer", "CompositeScorer",
        "MultiJudgeScorer", "LLMJudgeScorer", "LlamaGuardScorer",
        "PerspectiveAPIScorer", "RefusalRegexScorer",
        "KeyFactsScorer", "HedgePhraseScorer",
        # transforms
        "AttackTransform", "TranslationLaunderingTransform",
        "PersonaSwapTransform", "MultiTurnEscalationTransform",
        "ParaphraseTransform", "TyposTransform", "CaseChangeTransform",
        "IdentityTransform",
    }

    import argus
    missing = expected - set(dir(argus))
    assert not missing, f"missing from public surface: {sorted(missing)}"


def test_registries_are_populated():
    from argus import list_scorers, list_transforms

    scorers = list_scorers()
    transforms = list_transforms()
    # 10 concrete scorers (Phase 13 added presidio + exact_match_pii),
    # 7 transforms shipped in v2
    assert len(scorers) == 10, scorers
    assert len(transforms) == 7, transforms
    for required in (
        "llm_judge", "llama_guard", "composite", "multi_judge",
        "presidio", "exact_match_pii",
    ):
        assert required in scorers
    for required in ("identity", "translation_laundering", "persona_swap"):
        assert required in transforms
