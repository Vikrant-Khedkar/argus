import streamlit as st

from adapter import get_provider
from assistant import Assistant, load_persona
from risk_score import RiskScorer

st.set_page_config(page_title="Argus", page_icon="🛡️", layout="wide")

TIER_COLOR = {1: "🟢", 2: "🟡", 3: "🔴"}
AXIS_LABEL = {
    "output_liability": "Output Liability",
    "discrimination_liability": "Discrimination",
    "safety_liability": "Safety",
}


def score_emoji(score: float) -> str:
    if score >= 1.5:
        return "🟢"
    if score >= 1.0:
        return "🟡"
    return "🔴"


def build_assistant(provider_name: str, persona: str) -> Assistant:
    return Assistant(get_provider(provider_name), persona=persona)


if "provider_name" not in st.session_state:
    st.session_state.provider_name = "modal"
if "persona" not in st.session_state:
    st.session_state.persona = load_persona()
if "assistant" not in st.session_state:
    st.session_state.assistant = build_assistant(
        st.session_state.provider_name, st.session_state.persona
    )
if "risk_scorer" not in st.session_state:
    st.session_state.risk_scorer = RiskScorer()
if "risk_scores" not in st.session_state:
    st.session_state.risk_scores = {}
if "risk_panel_on" not in st.session_state:
    st.session_state.risk_panel_on = True
if "guardrails_on" not in st.session_state:
    st.session_state.guardrails_on = True
if "tool_use_on" not in st.session_state:
    st.session_state.tool_use_on = True

with st.sidebar:
    st.header("Argus 👁️")
    st.caption("the hundred-eyed AI risk audit")

    new_provider = st.radio(
        "Provider",
        options=["modal", "openrouter"],
        index=["modal", "openrouter"].index(st.session_state.provider_name),
        captions=["Qwen2.5-1.5B (OSS)", "Frontier via OpenRouter"],
    )
    if new_provider != st.session_state.provider_name:
        st.session_state.provider_name = new_provider
        st.session_state.assistant.set_provider(get_provider(new_provider))
        st.session_state.assistant.reset()
        st.session_state.risk_scores = {}
        st.toast(f"Switched to {new_provider} — chat reset")
        st.rerun()

    new_persona = st.text_area(
        "Persona / system prompt",
        value=st.session_state.persona,
        height=140,
    )
    if new_persona != st.session_state.persona:
        st.session_state.persona = new_persona
        st.session_state.assistant.set_persona(new_persona)

    if st.button("Reset conversation", use_container_width=True):
        st.session_state.assistant.reset()
        st.session_state.risk_scores = {}
        st.rerun()

    st.divider()
    st.session_state.risk_panel_on = st.toggle(
        "Live risk scoring",
        value=st.session_state.risk_panel_on,
        help="Judge each response with Claude Sonnet 4 on 3 liability axes. Adds ~5-10s latency per turn.",
    )
    st.session_state.guardrails_on = st.toggle(
        "Guardrails",
        value=st.session_state.guardrails_on,
        help="Pattern-block jailbreaks + post-flight regen on Tier 3 responses.",
    )
    st.session_state.tool_use_on = st.toggle(
        "Tool use (web search)",
        value=st.session_state.tool_use_on,
        help=(
            "Enable ScrapeGraphAI web_search tool. Both providers accept the tool schema; "
            "Qwen 1.5B invokes it ~20% of the time in our tests, GPT-4o-mini ~100%."
        ),
    )

    # Sync toggles to the live Assistant instance
    st.session_state.assistant.guardrails_enabled = st.session_state.guardrails_on
    st.session_state.assistant.tool_use_enabled = st.session_state.tool_use_on

    st.caption(
        f"Turns in memory: {len(st.session_state.assistant.conv.history) // 2}"
        + ("  •  summary present" if st.session_state.assistant.conv.summary else "")
    )

st.title("Chat")
st.caption("Generic assistant with live RiskScorer audit overlay")


def render_risk_badges(score):
    parts = [
        f"{TIER_COLOR[score.overall_tier]} **Tier {score.overall_tier}**",
        f"{score_emoji(score.output_liability)} output `{score.output_liability:.1f}`",
        f"{score_emoji(score.discrimination_liability)} discrim `{score.discrimination_liability:.1f}`",
        f"{score_emoji(score.safety_liability)} safety `{score.safety_liability:.1f}`",
    ]
    st.caption(" · ".join(parts))
    with st.expander("judge rationale", expanded=False):
        for axis in ("output_liability", "discrimination_liability", "safety_liability"):
            st.markdown(f"**{AXIS_LABEL[axis]}** — {score.rationale.get(axis, '')}")


for i, msg in enumerate(st.session_state.assistant.conv.history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and st.session_state.risk_panel_on:
            score = st.session_state.risk_scores.get(i)
            if score is not None:
                render_risk_badges(score)

if user_text := st.chat_input("Say something..."):
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        with st.spinner(f"{st.session_state.provider_name} thinking..."):
            try:
                reply = st.session_state.assistant.ask(user_text)
            except Exception as e:
                reply = f"[error] {e}"
        st.markdown(reply)

        if st.session_state.risk_panel_on and not reply.startswith("[error]"):
            # Reuse score from guardrails if it ran; else score now.
            score = st.session_state.assistant.last_risk
            if score is None:
                with st.spinner("Scoring risk with Claude Sonnet 4..."):
                    try:
                        score = st.session_state.risk_scorer.score(user_text, reply)
                    except Exception as e:
                        st.warning(f"Risk scoring failed: {e}")
                        score = None
            if score is not None:
                msg_idx = len(st.session_state.assistant.conv.history) - 1
                st.session_state.risk_scores[msg_idx] = score
                render_risk_badges(score)

    st.rerun()
