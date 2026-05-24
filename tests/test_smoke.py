"""Smoke tests for the Argus SDK surface.

Quick sanity check that imports + registries + canonical config builders
work. Run with:

    uv run python -m pytest tests/test_smoke.py -q
"""

from __future__ import annotations


def test_public_surface_resolves():
    expected = {
        # Core types
        "Instance", "ScoreResult", "RiskResult", "JudgeVerdict", "MultiJudgeResult",
        # Tier mapping
        "TierMapping", "InsuranceTierMapping", "RiskLevelMapping",
        "RawScoreMapping", "GradeLetterMapping",
        # Probes / axes
        "AxisSpec", "DEFAULT_AXES", "LiabilityProbe", "TransformedProbe",
        # Conversation
        "ConversationTurn", "ConversationProbe", "TurnResult", "ConversationResult",
        "score_conversation", "DEFAULT_MULTI_TURN_PROBES",
        # Providers
        "ChatProvider", "ModalProvider", "OpenRouterProvider",
        "HTTPProvider", "HuggingFaceProvider", "GuardrailedProvider",
        # Guardrails
        "PreFlightPatternGuard", "PreFlightClassifierGuard",
        "PreFlightEmbeddingGuard", "FailIndex", "refresh_fail_index",
        "PostFlightRegenGuard", "PostFlightHardRefuseGuard",
        # Scorers
        "Scorer", "ModelBasedScorer", "CompositeScorer", "MultiJudgeScorer",
        "LLMJudgeScorer", "LlamaGuardScorer", "LlamaPromptGuardScorer",
        "PerspectiveAPIScorer", "PresidioScorer", "ExactMatchPIIScorer",
        "RefusalRegexScorer", "KeyFactsScorer", "HedgePhraseScorer",
        # Transforms
        "AttackTransform", "TranslationLaunderingTransform",
        "PersonaSwapTransform", "MultiTurnEscalationTransform",
        "ParaphraseTransform", "TyposTransform", "CaseChangeTransform",
        "IdentityTransform",
        # Dataset loaders
        "load_factual_probes", "load_bias_probes", "load_safety_probes",
        "load_toxicity_probes", "load_calibration_probes", "load_pii_probes",
        # Storage / orchestration
        "AuditWriter", "AuditRow", "AuditIndex", "AuditReport",
        "EvalConfig", "Evaluator",
    }
    import argus
    missing = expected - set(dir(argus))
    assert not missing, f"missing from public surface: {sorted(missing)}"


def test_registries_populated():
    from argus import list_scorers, list_transforms

    scorers = list_scorers()
    transforms = list_transforms()
    # 11 concrete scorers, 7 transforms shipped
    assert len(scorers) == 11, scorers
    assert len(transforms) == 7, transforms
    for required in (
        "llm_judge", "llama_guard", "llama_prompt_guard", "composite",
        "multi_judge", "presidio", "exact_match_pii",
    ):
        assert required in scorers
    for required in ("identity", "translation_laundering", "persona_swap"):
        assert required in transforms


def test_eval_config_yaml_round_trip(tmp_path):
    import yaml
    from argus import EvalConfig

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "axes": {
            "safety_liability": {"type": "llama_guard"},
        },
        "transforms": ["identity"],
        "audit_log_path": str(tmp_path / "log.jsonl"),
        "audit_db_path": str(tmp_path / "db.sqlite"),
    }))
    cfg = EvalConfig.from_yaml(cfg_path)
    assert cfg.transforms == ["identity"]
    assert "safety_liability" in cfg.axes


def test_default_rubrics_load():
    from argus.rubrics import DEFAULT_RUBRICS
    assert "safety_liability" in DEFAULT_RUBRICS
    assert "output_liability" in DEFAULT_RUBRICS
    assert "discrimination_liability" in DEFAULT_RUBRICS
    assert "pii_leakage" in DEFAULT_RUBRICS
