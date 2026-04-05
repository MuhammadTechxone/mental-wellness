"""
HeedX AI - Nigerian Mental Wellness Assistant
Your personal mental wellness companion
"""

# ============================================
# IMPORTS
# ============================================
import html as html_lib
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import urllib.parse
import random
import requests

from config.settings import (
    QUESTIONNAIRE, MENTAL_HEALTH_CLASSES,
    RISK_THRESHOLDS, CONTACT_INFO,
    NIGERIA_EMERGENCY_NUMBERS, MENTAL_HEALTH_SUPPORT,
    PROFESSIONAL_HELP, FAITH_COMMUNITY_SUPPORT,
    NIGERIAN_PROVERBS, CULTURAL_WELLNESS_TIPS, CRISIS_RESOURCES
)
from utils.preprocessing import clean_response, split_sentences, preprocess_text
from utils.analysis import MentalHealthAnalyzer, create_distribution_dataframe
from utils.recommendations import RecommendationEngine
from utils.session_storage import SessionTracker

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="HeedX AI - Nigerian Mental Wellness",
    page_icon="NG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS  — professional, no emoji reliance
# ============================================
st.markdown("""
<style>
    /* ── Typography & base ── */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1rem;
    }

    /* ── Risk banners ── */
    .risk-high {
        background-color: #fff0f0;
        color: #1a1a1a;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #c0392b;
    }
    .risk-moderate {
        background-color: #fffbf0;
        color: #1a1a1a;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #e67e22;
    }
    .risk-low {
        background-color: #f0fff4;
        color: #1a1a1a;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #27ae60;
    }

    /* ── Sentence analysis box ── */
    .sentence-box {
        background-color: #f8f9fa;
        color: #1a1a1a;
        padding: 0.6rem 0.8rem;
        margin: 0.3rem 0;
        border-radius: 0.3rem;
        border-left: 3px solid #718096;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        font-size: 0.8rem;
        color: #718096;
        font-style: italic;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #e2e8f0;
    }

    /* ── Buttons ── */
    .stButton > button {
        width: 100%;
        background-color: #008751;
        color: white;
        font-weight: 600;
        border-radius: 0.4rem;
        border: none;
        padding: 0.5rem 1rem;
        transition: background-color 0.2s;
    }
    .stButton > button:hover {
        background-color: #006640;
    }

    /* ── Contact link buttons ── */
    .contact-button {
        display: block;
        text-align: center;
        padding: 0.45rem 1rem;
        margin: 0.2rem 0;
        background-color: #25D366;
        color: white;
        text-decoration: none;
        border-radius: 0.35rem;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .email-button {
        display: block;
        text-align: center;
        padding: 0.45rem 1rem;
        margin: 0.2rem 0;
        background-color: #c0392b;
        color: white;
        text-decoration: none;
        border-radius: 0.35rem;
        font-weight: 600;
        font-size: 0.88rem;
    }

    /* ── Nigerian flag accent bar ── */
    .nigeria-accent {
        background: linear-gradient(
            90deg,
            #008751 0%, #008751 33%,
            #ffffff 33%, #ffffff 66%,
            #008751 66%, #008751 100%
        );
        height: 4px;
        margin: 0.8rem 0 1.4rem 0;
        border-radius: 2px;
    }

    /* ── Chat bubbles ── */
    .chat-user {
        background-color: #008751;
        color: #ffffff;
        padding: 0.55rem 0.9rem;
        border-radius: 1rem 1rem 0.2rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.84rem;
        line-height: 1.5;
        text-align: right;
    }
    .chat-bot {
        background-color: #f1f5f9;
        color: #1a202c;
        padding: 0.55rem 0.9rem;
        border-radius: 1rem 1rem 1rem 0.2rem;
        margin: 0.4rem 0;
        font-size: 0.84rem;
        line-height: 1.6;
        border-left: 3px solid #008751;
    }
    .chat-system {
        background-color: #fef9e7;
        color: #7d6608;
        padding: 0.4rem 0.8rem;
        border-radius: 0.4rem;
        margin: 0.3rem 0;
        font-size: 0.8rem;
        border-left: 3px solid #f39c12;
    }

    /* ── Sidebar contact channel labels ── */
    .sidebar-channel {
        border-radius: 0.4rem;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.4rem;
        font-size: 0.83rem;
        font-weight: 600;
    }

    /* ── History card ── */
    .history-card {
        background: var(--secondary-background-color);
        padding: 0.5rem 0.8rem;
        border-radius: 0.4rem;
        margin: 0.3rem 0;
        border-left: 3px solid #008751;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# MODEL LOADING
# ============================================
@st.cache_resource
def load_model():
    """Load the pre-trained SVC pipeline model"""
    model_path = 'psychotherapy_svm_model.pkl'

    if not os.path.exists(model_path):
        st.error("Model file not found: " + model_path)
        st.info("Please ensure 'psychotherapy_svm_model.pkl' is in the app directory.")
        return None

    try:
        model = joblib.load(model_path)
        with st.sidebar.expander("Model Info", expanded=False):
            st.write("**Model type:** " + type(model).__name__)
            st.write("**Classes:** " + ", ".join(model.classes_))
        return model
    except Exception as e:
        st.error("Error loading model: " + str(e))
        return None


# ============================================
# SESSION STATE INITIALIZATION
# ============================================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'page':              'home',
        'responses':         {},
        'analysis_complete': False,
        'analyzer':          None,
        'results':           None,
        'tracker':           None,
        'chat_messages':     [],
        'groq_api_key':      '',
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================
# GROQ API CALL
# ============================================
def call_groq(messages: list, api_key: str, temperature: float = 0.7) -> str:
    """
    Call Groq's Llama-3.3-70b-versatile model.
    Returns the assistant reply text, or an error string.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       "llama-3.3-70b-versatile",
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  600,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        return "The response took too long. Please try again."
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response else "?"
        if code == 401:
            return "Invalid Groq API key. Please check your key in the sidebar."
        if code == 429:
            return "Rate limit reached. Please wait a moment and try again."
        return "API error (" + str(code) + "). Please try again."
    except Exception as e:
        return "Could not reach the AI service. Please check your connection."


# ============================================
# SESSION CONTEXT BUILDER
# ============================================
def build_session_context() -> str:
    """
    Build a structured context string from the user's current session:
    - SVC model assessment results
    - Risk level
    - Distribution percentages
    - Progress tracking trends
    Injected into the chatbot system prompt so it speaks from knowledge,
    not guesswork.
    """
    parts = []

    # Assessment results from the SVC model
    results  = st.session_state.get('results')
    analyzer = st.session_state.get('analyzer')

    if results and analyzer:
        parts.append("CURRENT SESSION ASSESSMENT (from SVC psychotherapy model):")
        parts.append(
            "  Primary indicator : " + results['primary_indicator'].upper() +
            " (" + f"{results['primary_percentage']:.1f}" + "%)"
        )
        if results.get('secondary_indicator'):
            parts.append(
                "  Secondary indicator: " + results['secondary_indicator'].upper() +
                " (" + f"{results['secondary_percentage']:.1f}" + "%)"
            )
        parts.append("  Full distribution:")
        for cls, pct in results['distribution'].items():
            parts.append("    - " + cls + ": " + f"{pct:.1f}" + "%")

        risk_level = analyzer.get_risk_level()
        parts.append("  Risk level: " + risk_level.upper())

        flagged = analyzer.get_suicidal_flagged_sentences()
        if flagged:
            parts.append(
                "  Note: " + str(len(flagged)) +
                " sentence(s) flagged for suicidal ideation indicators."
            )
    else:
        parts.append("ASSESSMENT STATUS: The user has not yet completed a check-in this session.")

    # Progress tracking trends
    tracker = st.session_state.get('tracker')
    if tracker:
        summary = tracker.get_progress_summary()
        if summary and summary.get('total_assessments', 0) > 0:
            parts.append("\nSESSION PROGRESS TRACKING:")
            parts.append("  Total check-ins this session: " + str(summary['total_assessments']))
            parts.append("  Most common primary indicator: " + str(summary.get('most_common_primary', 'N/A')))
            trend = summary.get('trend', {})
            if trend:
                parts.append("  Trends since first check-in:")
                for cls, delta in trend.items():
                    direction = "up" if delta > 0 else "down"
                    parts.append(
                        "    - " + cls + ": " + direction + " " + f"{abs(delta):.1f}" + "%"
                    )

    return "\n".join(parts) if parts else "No session data available yet."


# ============================================
# SYSTEM PROMPT BUILDER
# ============================================
def build_system_prompt() -> str:
    """
    Construct the full system prompt for the Groq model.
    Includes the HeedX persona, Nigerian cultural context,
    professional boundaries, and the user's live session data.
    """
    session_context = build_session_context()
    has_assessment  = st.session_state.get('results') is not None

    prompt = """You are HeedX, a professional AI mental wellness assistant embedded in the HeedX platform — a Nigerian mental health support application. You were developed to provide empathetic, culturally aware, and clinically informed mental wellness conversations for Nigerian users.

YOUR ROLE:
- Provide supportive, evidence-informed mental wellness conversations
- Give specific, actionable recommendations and coping strategies
- Interpret and explain the user's SVC model assessment results in plain language
- Reference the user's progress trends when relevant
- Maintain professional, warm, and non-judgmental tone throughout
- Be culturally sensitive to Nigerian values, family structures, and community dynamics

CLINICAL BOUNDARIES:
- You are not a licensed therapist and do not diagnose
- For high-risk or suicidal indicators, always direct to Nigerian crisis lines: 112 (emergency), 08062106493 (MANI), 08099769974 (She Writes Woman)
- Encourage professional consultation when warranted
- Never dismiss or minimise distress

TONE AND STYLE:
- Professional but warm — like a knowledgeable friend with clinical training
- Do not use emojis
- Write in clear, plain English that is accessible to Nigerian users
- Responses should be conversational but substantive — typically 3 to 6 sentences
- Vary your responses; do not repeat the same phrases

NIGERIAN CULTURAL CONTEXT:
- Acknowledge the role of faith, family, and community in Nigerian mental health
- Be aware of stigma around mental health in Nigerian society
- Reference local support systems where appropriate (Federal Neuro Hospitals: Yaba, Kaduna, Aro-Abeokuta)

"""

    if has_assessment:
        prompt += "USER SESSION DATA (from the HeedX SVC psychotherapy model — use this to personalise your responses):\n"
        prompt += session_context
        prompt += "\n\nWhen the user asks about their results, feelings, or progress, refer to the above data specifically. Do not ask them to describe what the model already shows — interpret it for them and build on it."
    else:
        prompt += "USER SESSION DATA:\n"
        prompt += session_context
        prompt += "\n\nThe user has not completed a check-in yet. Warmly encourage them to complete one using the Check-In page so you can provide personalised insights. You may still engage in general mental wellness conversation in the meantime."

    return prompt


# ============================================
# GROQ CHATBOT RESPONSE
# ============================================
def get_ai_response(user_message: str, api_key: str) -> str:
    """
    Build the full message history and call Groq.
    Injects a fresh system prompt on every call so the
    context always reflects the latest session state.
    """
    system_prompt = build_system_prompt()

    # Crisis shortcut — handled locally before hitting API
    crisis_keywords = [
        'kill myself', 'end my life', 'suicide', 'want to die',
        'no reason to live', 'hang myself', 'overdose',
        'hurt myself', 'self harm', 'take my life'
    ]
    if any(kw in user_message.lower() for kw in crisis_keywords):
        return (
            "What you have just shared is serious and I am glad you are talking. "
            "Please reach out for immediate support right now:\n\n"
            "- Emergency: Call 112\n"
            "- Mentally Aware Nigeria Initiative (MANI): 08062106493\n"
            "- She Writes Woman: 08099769974\n"
            "- Visit the nearest Federal Neuropsychiatric Hospital\n\n"
            "You do not have to face this alone. Would you like to talk about what is happening?"
        )

    # Build message list: system + conversation history + new user message
    messages = [{"role": "system", "content": system_prompt}]

    # Include up to last 10 exchanges for context window efficiency
    for msg in st.session_state.chat_messages[-10:]:
        messages.append({
            "role":    "user"      if msg['role'] == 'user' else "assistant",
            "content": msg['content']
        })

    messages.append({"role": "user", "content": user_message})

    return call_groq(messages, api_key)


# ============================================
# CHAT PAGE  (full page, mobile-friendly)
# ============================================
def render_chat_page():
    """
    Full-page Groq Llama 3 mental wellness chatbot.
    Context-aware of SVC assessment results, risk level and session progress.
    API key is loaded from .streamlit/secrets.toml — users never need to enter one.
    """
    st.markdown("<h2 class='sub-header'>Wellness Chat</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#4a5568;margin-top:-0.5rem;margin-bottom:1rem;'>"
        "Speak with the HeedX AI wellness assistant. Responses are informed by "
        "your assessment results and session progress.</p>",
        unsafe_allow_html=True
    )

    # Resolve API key — secrets.toml takes priority, no user input needed
    if not st.session_state.groq_api_key:
        try:
            st.session_state.groq_api_key = st.secrets["GROQ_API_KEY"]
        except (KeyError, FileNotFoundError):
            st.session_state.groq_api_key = ""

    if not st.session_state.groq_api_key:
        st.error(
            "The chat assistant is not configured. "
            "Please add GROQ_API_KEY to .streamlit/secrets.toml and restart the app."
        )
        if st.button("Back to Home", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
        return

    # Context status banner
    has_assessment = st.session_state.get('results') is not None
    tracker        = st.session_state.get('tracker')
    has_progress   = tracker is not None and tracker.has_history()

    if has_assessment:
        results  = st.session_state['results']
        primary  = results['primary_indicator'].upper()
        pct      = f"{results['primary_percentage']:.1f}"
        risk     = st.session_state['analyzer'].get_risk_level().upper()
        checkins = len(tracker.assessments) if has_progress else 0
        st.markdown(
            "<div style='background:#f0faf4;border-left:4px solid #008751;"
            "border-radius:0 0.5rem 0.5rem 0;padding:0.75rem 1rem;"
            "margin-bottom:1rem;font-size:0.88rem;color:#1a1a1a;'>"
            "<strong>Active context from your assessment</strong><br>"
            "Primary indicator: <strong>" + primary + "</strong> (" + pct + "%)&nbsp;&nbsp;|&nbsp;&nbsp;"
            "Risk level: <strong>" + risk + "</strong>&nbsp;&nbsp;|&nbsp;&nbsp;"
            "Check-ins: <strong>" + str(checkins) + "</strong>"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background:#fef9e7;border-left:4px solid #e67e22;"
            "border-radius:0 0.5rem 0.5rem 0;padding:0.75rem 1rem;"
            "margin-bottom:1rem;font-size:0.88rem;color:#7d4e00;'>"
            "No assessment completed yet. The assistant will encourage you to complete "
            "a check-in so it can provide personalised responses based on your results."
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Chat history
    if not st.session_state.chat_messages:
        st.markdown(
            "<div class='chat-bot' style='max-width:640px;'>"
            "Good day. I am the HeedX wellness assistant, powered by Llama 3. "
            "I have access to your session data and assessment results to provide "
            "personalised mental wellness support. How are you feeling today?"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        for msg in st.session_state.chat_messages:
            if msg['role'] == 'user':
                st.markdown(
                    "<div style='display:flex;justify-content:flex-end;margin:0.4rem 0;'>"
                    "<div class='chat-user' style='max-width:80%;'>"
                    + html_lib.escape(msg['content']) +
                    "</div></div>",
                    unsafe_allow_html=True
                )
            else:
                content = msg['content'].replace('\n', '<br>')
                st.markdown(
                    "<div style='display:flex;justify-content:flex-start;margin:0.4rem 0;'>"
                    "<div class='chat-bot' style='max-width:85%;'>"
                    + content +
                    "</div></div>",
                    unsafe_allow_html=True
                )

    st.markdown("---")

    # Input form
    with st.form("chat_page_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your message",
            placeholder="Share how you are feeling, ask a question, or discuss your results...",
            height=110,
            label_visibility="collapsed",
            key="chat_page_input"
        )
        col_send, col_clear, col_back = st.columns([3, 1, 1])
        with col_send:
            send = st.form_submit_button("Send", use_container_width=True)
        with col_clear:
            clear = st.form_submit_button("Clear chat", use_container_width=True)
        with col_back:
            back = st.form_submit_button("Back", use_container_width=True)

    if send and user_input and user_input.strip():
        user_text = user_input.strip()
        st.session_state.chat_messages.append({'role': 'user', 'content': user_text})
        with st.spinner("HeedX is responding..."):
            reply = get_ai_response(user_text, st.session_state.groq_api_key)
        st.session_state.chat_messages.append({'role': 'bot', 'content': reply})
        st.rerun()

    if clear:
        st.session_state.chat_messages = []
        st.rerun()

    if back:
        st.session_state.page = 'home'
        st.rerun()

    st.caption(
        "Powered by Llama 3.3-70b via Groq. Responses are supportive in nature "
        "and do not constitute professional medical advice."
    )


# ============================================
# SIDEBAR CONTACT CHANNELS
# ============================================
def render_sidebar_contacts():
    """Quick-access contact buttons in the sidebar"""

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Contact & Support")

    # Counselling & Guidance
    st.sidebar.markdown(
        "<div class='sidebar-channel' style='background:#f0faf4;border-left:3px solid #008751;'>"
        "<span style='color:#008751;'>Counselling and Guidance</span>"
        "</div>",
        unsafe_allow_html=True
    )

    whatsapp_number = CONTACT_INFO['whatsapp'].lstrip('0')
    whatsapp_link   = "https://wa.me/234" + whatsapp_number
    call_number     = CONTACT_INFO.get('phone', CONTACT_INFO['whatsapp'])

    st.sidebar.markdown(
        "<a href='" + whatsapp_link + "' target='_blank' class='contact-button'>"
        "WhatsApp Counsellor</a>",
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        "<a href='tel:" + call_number + "' class='contact-button' "
        "style='background-color:#008751;margin-bottom:0.6rem;'>"
        "Call " + call_number + "</a>",
        unsafe_allow_html=True
    )

    # Platform Support
    st.sidebar.markdown(
        "<div class='sidebar-channel' style='background:#f0f4ff;border-left:3px solid #2980b9;'>"
        "<span style='color:#1565C0;'>Platform Support</span>"
        "</div>",
        unsafe_allow_html=True
    )

    support_email = CONTACT_INFO.get('support_email', CONTACT_INFO['email'])
    st.sidebar.markdown(
        "<a href='mailto:" + support_email + "' target='_blank' class='email-button'>"
        "Email Platform Support</a>",
        unsafe_allow_html=True
    )

    platform_wa = CONTACT_INFO.get('platform_whatsapp', '')
    if platform_wa:
        pw_link = "https://wa.me/234" + platform_wa.lstrip('0')
        st.sidebar.markdown(
            "<a href='" + pw_link + "' target='_blank' class='contact-button'>"
            "Platform WhatsApp</a>",
            unsafe_allow_html=True
        )


# ============================================
# NIGERIAN SUPPORT RESOURCES
# ============================================
def render_nigerian_support():
    """Display comprehensive Nigerian support resources"""

    with st.expander("Nigerian Support Resources", expanded=False):
        st.markdown("### " + NIGERIA_EMERGENCY_NUMBERS['national']['title'])
        for resource in NIGERIA_EMERGENCY_NUMBERS['national']['resources']:
            st.markdown("- " + resource)

        st.markdown("### " + NIGERIA_EMERGENCY_NUMBERS['state_emergency']['title'])
        for resource in NIGERIA_EMERGENCY_NUMBERS['state_emergency']['resources']:
            st.markdown("- " + resource)

        st.markdown("---")
        st.markdown("### " + MENTAL_HEALTH_SUPPORT['title'])
        for resource in MENTAL_HEALTH_SUPPORT['resources']:
            st.markdown("- " + resource)

        st.markdown("---")
        st.markdown("### " + PROFESSIONAL_HELP['title'])
        for resource in PROFESSIONAL_HELP['resources']:
            st.markdown("- " + resource)

        st.markdown("---")
        st.markdown("### " + FAITH_COMMUNITY_SUPPORT['title'])
        for resource in FAITH_COMMUNITY_SUPPORT['resources']:
            st.markdown("- " + resource)

        st.markdown("---")
        st.markdown("### Nigerian Wisdom for Mental Wellness")
        for proverb in NIGERIAN_PROVERBS[:4]:
            st.markdown("*\"" + proverb + "\"*")

        st.markdown("---")
        st.markdown("**Remember:** Help is available. You are not alone.")


# ============================================
# FEEDBACK / CONTACT SECTION (main page)
# ============================================
def render_feedback_section():
    """Two clearly separated contact channels on the main page"""

    st.markdown("### Contact and Support")

    # Counselling & Guidance
    st.markdown("""
    <div style="background:#f0faf4;border-left:4px solid #008751;
                border-radius:0 0.5rem 0.5rem 0;padding:1rem 1.2rem;margin-bottom:1rem;">
        <p style="margin:0 0 0.3rem 0;font-weight:700;color:#008751;font-size:0.95rem;">
            Mental Wellness Counselling and Guidance
        </p>
        <p style="margin:0;font-size:0.88rem;color:#2d3748;">
            Reach out if you need to talk, seek emotional guidance, or want
            personalised wellness advice from our counsellor.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**WhatsApp — Counselling**")
        whatsapp_number = CONTACT_INFO['whatsapp'].lstrip('0')
        whatsapp_link   = "https://wa.me/234" + whatsapp_number
        st.markdown(
            "<a href='" + whatsapp_link + "' target='_blank' class='contact-button'>"
            "Chat with Counsellor</a>",
            unsafe_allow_html=True
        )
        st.caption("Monday to Friday, 9 am to 5 pm")

    with col2:
        st.markdown("**Call or SMS**")
        call_number = CONTACT_INFO.get('phone', CONTACT_INFO['whatsapp'])
        st.markdown(
            "<a href='tel:" + call_number + "' class='contact-button' "
            "style='background-color:#008751;'>"
            "Call " + call_number + "</a>",
            unsafe_allow_html=True
        )
        st.caption("Voice calls welcome during hours above")

    st.markdown("<br>", unsafe_allow_html=True)

    # Platform Support
    st.markdown("""
    <div style="background:#f0f4ff;border-left:4px solid #2980b9;
                border-radius:0 0.5rem 0.5rem 0;padding:1rem 1.2rem;margin-bottom:1rem;">
        <p style="margin:0 0 0.3rem 0;font-weight:700;color:#1565C0;font-size:0.95rem;">
            Platform Support
        </p>
        <p style="margin:0;font-size:0.88rem;color:#2d3748;">
            Bug reports, feature requests, account questions, or any technical
            issue with the HeedX AI platform.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Support Email**")
        support_email = CONTACT_INFO.get('support_email', CONTACT_INFO['email'])
        st.markdown(
            "<a href='mailto:" + support_email + "' target='_blank' class='email-button'>"
            "Email Platform Support</a>",
            unsafe_allow_html=True
        )
        st.caption("Response within 24 to 48 hours")

    with col4:
        st.markdown("**Platform WhatsApp**")
        platform_wa = CONTACT_INFO.get('platform_whatsapp', CONTACT_INFO.get('whatsapp_support', ''))
        if platform_wa:
            pw_link = "https://wa.me/234" + platform_wa.lstrip('0')
            st.markdown(
                "<a href='" + pw_link + "' target='_blank' class='contact-button'>"
                "Platform WhatsApp</a>",
                unsafe_allow_html=True
            )
        else:
            st.info("Platform WhatsApp — coming soon")

    with st.expander("Send a Message", expanded=False):
        contact_type = st.radio(
            "What is this about?",
            ["Counselling and Guidance", "Platform Support, Bug or Feature Request"],
            horizontal=True
        )
        feedback_type = st.selectbox(
            "Category",
            ["General Enquiry", "Suggestion", "Bug Report",
             "Feature Request", "Support Question", "Other"]
        )
        feedback_text = st.text_area(
            "Your Message",
            placeholder="Please describe your question or feedback...",
            height=100
        )
        if st.button("Send Message"):
            if feedback_text.strip():
                subject     = "HeedX AI - " + contact_type.split()[0] + ": " + feedback_type
                body        = "Category: " + feedback_type + "\nChannel: " + contact_type + "\n\nMessage:\n" + feedback_text
                recipient   = CONTACT_INFO['email'] if "Counselling" in contact_type else CONTACT_INFO.get('support_email', CONTACT_INFO['email'])
                mailto_link = "mailto:" + recipient + "?subject=" + urllib.parse.quote(subject) + "&body=" + urllib.parse.quote(body)
                st.markdown(
                    "<a href='" + mailto_link + "' target='_blank'>Click to open email client</a>",
                    unsafe_allow_html=True
                )
                st.success("Email client opened. Please send the pre-filled email.")
            else:
                st.warning("Please enter a message before sending.")


# ============================================
# PROGRESS TRACKING
# ============================================
def render_progress_tracking():
    """Render user progress tracking visualizations"""

    st.markdown("""
    <div style="margin:2rem 0 1rem 0;text-align:center;">
        <h3 style="color:#008751;font-weight:600;">Your Progress This Session</h3>
        <p style="color:#4a5568;margin-top:0.4rem;">
            Track your mental wellness journey and notice patterns over time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get('tracker'):
        st.info("Complete a check-in to start tracking your progress this session.")
        return

    tracker    = st.session_state.tracker
    df_history = tracker.get_history_dataframe()

    if df_history.empty:
        st.info("No assessments yet. Complete a check-in to see your progress.")
        return

    summary = tracker.get_progress_summary()

    # Summary banner
    st.markdown(
        "<div style='background:linear-gradient(135deg,#e6fffa 0%,#c6f6d5 100%);"
        "border-radius:1rem;padding:1.2rem;text-align:center;"
        "margin-bottom:1.5rem;border:1px solid #9ae6b4;'>"
        "<span style='font-size:1rem;color:#1a202c;'>You have completed </span>"
        "<span style='font-size:2rem;font-weight:700;color:#065f46;margin:0 0.5rem;'>"
        + str(len(df_history)) +
        "</span>"
        "<span style='font-size:1rem;color:#1a202c;'>"
        + ("check-ins" if len(df_history) > 1 else "check-in") +
        " in this session.</span>"
        "</div>",
        unsafe_allow_html=True
    )

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    def metric_card(value, label):
        return (
            "<div style='background:white;border-radius:0.75rem;padding:1rem;"
            "text-align:center;box-shadow:0 1px 6px rgba(0,0,0,0.06);"
            "border:1px solid #e2e8f0;'>"
            "<div style='font-size:1.4rem;font-weight:700;color:#008751;'>" + str(value) + "</div>"
            "<div style='color:#4a5568;font-size:0.78rem;margin-top:0.2rem;'>" + label + "</div>"
            "</div>"
        )

    with col1:
        st.markdown(metric_card(summary['total_assessments'], "Total Check-ins"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card(summary['first_date'], "First Check-in"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card(summary['last_date'], "Most Recent"), unsafe_allow_html=True)
    with col4:
        most_common = summary.get('most_common_primary', 'N/A')
        st.markdown(metric_card(most_common.title() if most_common else "N/A", "Most Common"), unsafe_allow_html=True)

    # Trend chart
    st.markdown("#### Indicator Trends")

    fig = go.Figure()
    traces = [
        ("Anxiety",    "#e67e22", "anxiety"),
        ("Depression", "#2980b9", "depression"),
        ("Suicidal",   "#c0392b", "suicidal"),
        ("Normal",     "#27ae60", "normal"),
    ]
    for name, color, col in traces:
        if col in df_history.columns:
            fig.add_trace(go.Scatter(
                x=df_history['assessment'],
                y=df_history[col],
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=2.5),
                marker=dict(size=8),
            ))

    fig.update_layout(
        xaxis=dict(title="Check-in Number"),
        yaxis=dict(title="Percentage (%)", range=[0, 100]),
        hovermode='x unified',
        height=420,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    st.plotly_chart(fig, use_container_width=True)

    # Trend insights
    if summary.get('trend'):
        trend    = summary['trend']
        insights = []

        if trend.get('normal', 0) > 5:
            insights.append("Positive wellbeing improvement of +" + str(round(trend['normal'])) + "% since your first check-in.")
        if trend.get('anxiety', 0) < 0:
            insights.append("Anxiety indicators have reduced by " + str(abs(round(trend['anxiety']))) + "%.")
        if trend.get('depression', 0) < 0:
            insights.append("Depression indicators are improving by " + str(abs(round(trend['depression']))) + "%.")
        if trend.get('suicidal', 0) > 2:
            insights.append("An increase in difficult thoughts has been noted. Please consider reaching out for support.")
        elif trend.get('suicidal', 0) > 0:
            insights.append("A slight increase in difficult thought indicators. Continued support is recommended.")
        if not insights:
            insights.append("Your indicators are stable across this session.")

        st.markdown("**Session Insights**")
        for insight in insights[:4]:
            st.markdown("- " + insight)

    with st.expander("View All Check-Ins"):
        st.dataframe(df_history, use_container_width=True)

    st.caption("Your data exists only within this browser session and is cleared when you close the tab.")


# ============================================
# HOME PAGE
# ============================================
def render_home_page():
    """Render the home page"""

    st.markdown("<h1 class='main-header'>HeedX AI</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#008751;font-size:1.05rem;'>"
        "Your Nigerian Mental Wellness Companion</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center;color:#4a5568;font-size:0.9rem;'>"
        "Understand. Respond. Support.</p>",
        unsafe_allow_html=True
    )
    st.markdown("<div class='nigeria-accent'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="background:white;border-radius:0.75rem;padding:1.5rem;margin:1rem 0;
                    text-align:center;box-shadow:0 2px 10px rgba(0,0,0,0.05);
                    border:1px solid #e2e8f0;">
            <p style="font-size:1.1rem;color:#2d3748;margin:0;">
                A professional space to reflect, understand, and nurture your mental wellbeing.
            </p>
            <p style="font-size:0.88rem;color:#008751;margin-top:0.6rem;margin-bottom:0;">
                Nigerian resources &nbsp;|&nbsp; Culturally adapted &nbsp;|&nbsp; Free and confidential
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ### How It Works

        1. **Complete a check-in** by answering questions about how you have been feeling
        2. **The AI model analyzes** your responses at sentence level
        3. **Receive insights**, risk assessment, and tailored recommendations
        4. **Chat with HeedX**, our AI wellness assistant, for personalised guidance
        5. **Track your journey** and monitor changes across check-ins

        All data remains private within your current browser session.
        """)

        render_nigerian_support()

        if st.session_state.tracker:
            st.info(
                "Check-ins completed this session: " + str(len(st.session_state.tracker.assessments))
            )

        if st.button("Start Your Check-In", use_container_width=True):
            if not st.session_state.tracker:
                st.session_state.tracker = SessionTracker()
            st.session_state.page = 'questionnaire'
            st.rerun()

        if st.session_state.tracker and st.session_state.tracker.has_history():
            st.markdown("---")
            st.markdown("### Recent Session Activity")
            history_df = st.session_state.tracker.get_history_dataframe()
            if not history_df.empty:
                for _, row in history_df.tail(3).iterrows():
                    st.markdown(
                        "<div class='history-card'>"
                        + str(row['date']) +
                        " &nbsp;&mdash;&nbsp; Primary: " + str(row['primary']).upper() +
                        " &nbsp;&mdash;&nbsp; " + str(round(row['primary_pct'])) + "%"
                        "</div>",
                        unsafe_allow_html=True
                    )
                if st.button("View Full Progress", use_container_width=True):
                    st.session_state.page = 'progress'
                    st.rerun()

        st.markdown("""
        <div class='disclaimer'>
            <p>This is a supportive tool, not a medical diagnosis.</p>
            <p>If you are in crisis, call <strong>112</strong> (National Emergency)
               or <strong>08062106493</strong> (Mentally Aware Nigeria Initiative) immediately.</p>
            <p>Your life is valuable. Help is available.</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================
# QUESTIONNAIRE PAGE
# ============================================
def render_questionnaire_page():
    """Render the questionnaire page"""

    if not st.session_state.tracker:
        st.session_state.tracker = SessionTracker()

    st.markdown("<h2 class='sub-header'>Mental Health Questionnaire</h2>",
                unsafe_allow_html=True)

    if len(st.session_state.tracker.assessments) > 0:
        st.info("Check-ins completed this session: " + str(len(st.session_state.tracker.assessments)))

    st.markdown(
        "Answer each question as fully as you can. "
        "The more detail you provide, the more accurate the analysis will be."
    )

    if 'responses' not in st.session_state:
        st.session_state.responses = {}

    with st.form("questionnaire_form"):
        for q in QUESTIONNAIRE:
            st.markdown("**" + q['question'] + "**")
            default_value = st.session_state.get("input_" + q['id'], "")
            response = st.text_area(
                label="Response for " + q['id'],
                label_visibility="collapsed",
                placeholder=q['placeholder'],
                value=default_value,
                height=100,
                key="input_" + q['id']
            )
            st.session_state.responses[q['id']] = response
            st.markdown("---")

        submitted = st.form_submit_button("Analyze My Check-In", use_container_width=True)

        if submitted:
            has_content = any(v and v.strip() for v in st.session_state.responses.values())
            if has_content:
                st.session_state.page = 'analysis'
                st.rerun()
            else:
                st.warning("Please share a little about how you are feeling before submitting.")

    col1, _, _ = st.columns([1, 2, 1])
    with col1:
        if st.button("Back to Home", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()


# ============================================
# ANALYSIS PAGE
# ============================================
def render_analysis_page(model):
    """Render the analysis page with sentence-level results"""

    st.markdown("<h2 class='sub-header'>Analyzing Your Responses</h2>",
                unsafe_allow_html=True)

    analyzer      = MentalHealthAnalyzer(model, model.classes_)
    progress_bar  = st.progress(0)
    status_text   = st.empty()
    all_sentences = []
    errors        = 0

    for i, (q_id, response) in enumerate(st.session_state.responses.items()):
        if response and response.strip():
            status_text.text("Analyzing question " + str(i + 1) + " of " + str(len(QUESTIONNAIRE)) + "...")
            try:
                cleaned          = clean_response(response)
                sentence_results = analyzer.analyze_response(cleaned, q_id)
                all_sentences.extend(sentence_results)
            except Exception:
                errors += 1
                st.warning("Could not fully analyze question " + str(i + 1) + " — skipping.")

        progress_bar.progress((i + 1) / len(QUESTIONNAIRE))

    if errors:
        st.info(str(errors) + " question(s) were skipped due to processing issues.")

    st.session_state.analyzer          = analyzer
    analyzer.sentence_predictions      = all_sentences
    analyzer.aggregated_results        = analyzer._aggregate_predictions()
    st.session_state.results           = analyzer.aggregated_results
    st.session_state.analysis_complete = True

    if st.session_state.tracker:
        st.session_state.tracker.add_assessment(
            analyzer.aggregated_results,
            st.session_state.responses
        )

    status_text.text("Analysis complete.")
    progress_bar.empty()

    st.markdown("### Sentence-Level Analysis")
    st.caption(
        "Each sentence from your responses has been classified by the psychotherapy model. "
        "Results below show the predicted indicator per sentence."
    )

    COLOR_MAP = {
        'suicidal':   ("#fff0f0", "Suicidal"),
        'anxiety':    ("#fffbf0", "Anxiety"),
        'depression': ("#f0f4ff", "Depression"),
        'normal':     ("#f8f9fa", "Normal"),
    }

    visible = all_sentences[:10]
    hidden  = all_sentences[10:]

    if visible:
        for sent_result in visible:
            prediction    = sent_result.get('prediction', 'normal')
            color, label  = COLOR_MAP.get(prediction, ("#f8f9fa", "Normal"))
            safe_sentence = html_lib.escape(sent_result.get('sentence', ''))
            conf_text     = ""
            if sent_result.get('confidence'):
                conf_text = " &mdash; Confidence: " + f"{sent_result['confidence']:.1%}"

            html_block = (
                "<div class='sentence-box' style='background-color:" + color + ";'>"
                "<span style='color:#555;font-size:0.82rem;'>" + label + conf_text + "</span><br>"
                "<span>" + safe_sentence + "</span>"
                "</div>"
            )
            st.markdown(html_block, unsafe_allow_html=True)

        if hidden:
            with st.expander("Show remaining " + str(len(hidden)) + " sentences"):
                for sent_result in hidden:
                    prediction    = sent_result.get('prediction', 'normal')
                    color, label  = COLOR_MAP.get(prediction, ("#f8f9fa", "Normal"))
                    safe_sentence = html_lib.escape(sent_result.get('sentence', ''))
                    conf_text     = ""
                    if sent_result.get('confidence'):
                        conf_text = " &mdash; Confidence: " + f"{sent_result['confidence']:.1%}"
                    html_block = (
                        "<div class='sentence-box' style='background-color:" + color + ";'>"
                        "<span style='color:#555;font-size:0.82rem;'>" + label + conf_text + "</span><br>"
                        "<span>" + safe_sentence + "</span>"
                        "</div>"
                    )
                    st.markdown(html_block, unsafe_allow_html=True)
    else:
        st.warning("No analyzable sentences were found. Please provide more detail in your responses.")

    if st.button("View Assessment Summary", use_container_width=True):
        st.session_state.page = 'summary'
        st.rerun()


# ============================================
# SUMMARY PAGE
# ============================================
def render_summary_page():
    """Render the summary page with results and progress"""

    st.markdown("<h2 class='sub-header'>Assessment Summary</h2>",
                unsafe_allow_html=True)

    if not st.session_state.results:
        st.error("No analysis results found.")
        if st.button("Back to Questionnaire"):
            st.session_state.page = 'questionnaire'
            st.rerun()
        return

    results          = st.session_state.results
    analyzer         = st.session_state.analyzer
    risk_level       = analyzer.get_risk_level()
    recommender      = RecommendationEngine()
    suicidal_flagged = analyzer.get_suicidal_flagged_sentences()
    recommendations  = recommender.get_recommendations(
        results['primary_indicator'], risk_level, suicidal_flagged
    )

    # Indicators
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Primary Indicator")
        primary = results['primary_indicator']
        pct     = results['primary_percentage']
        if primary == 'suicidal':
            st.error("**" + primary.upper() + "** (" + f"{pct:.1f}" + "%)")
        elif primary in ['anxiety', 'depression']:
            st.warning("**" + primary.upper() + "** (" + f"{pct:.1f}" + "%)")
        else:
            st.success("**" + primary.upper() + "** (" + f"{pct:.1f}" + "%)")

    with col2:
        if results['secondary_indicator']:
            st.markdown("### Secondary Indicator")
            secondary = results['secondary_indicator']
            st.info("**" + secondary.upper() + "** (" + f"{results['secondary_percentage']:.1f}" + "%)")

    # Risk level
    st.markdown("### Risk Assessment")
    risk_map = {
        "high":          ("risk-high",     "HIGH RISK — Immediate support is strongly recommended."),
        "moderate_high": ("risk-high",     "MODERATE-HIGH RISK — Please speak with a mental health professional."),
        "moderate":      ("risk-moderate", "MODERATE RISK — Consider professional support."),
        "mild":          ("risk-moderate", "MILD RISK — Self-care strategies are recommended."),
        "low":           ("risk-low",      "LOW RISK — Continue maintaining your mental wellness."),
    }
    css_class, message = risk_map.get(risk_level, ("risk-low", "Status could not be determined."))
    st.markdown("<div class='" + css_class + "'>" + message + "</div>", unsafe_allow_html=True)

    # Distribution chart
    st.markdown("### Distribution of Indicators")
    df_dist = create_distribution_dataframe(results['distribution'])
    fig = px.bar(
        df_dist,
        x='Mental Health Indicator',
        y='Percentage',
        color='Mental Health Indicator',
        color_discrete_map={
            'anxiety':    '#e67e22',
            'depression': '#2980b9',
            'suicidal':   '#c0392b',
            'normal':     '#27ae60',
        },
        text=df_dist['Percentage'].apply(lambda x: f"{x:.1f}%")
    )
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Percentage of Sentences (%)",
        showlegend=False,
        height=380,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=10, b=40)
    )
    fig.update_traces(textposition='outside')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    st.markdown("### Recommendations")
    if recommendations:
        st.markdown("#### " + recommendations['title'])
        st.markdown(recommendations['message'])
        for i, tip in enumerate(recommendations['tips'], 1):
            st.markdown(str(i) + ". " + tip)

    # Crisis resources
    if risk_level in ['high', 'moderate_high']:
        with st.expander("Immediate Help — Nigeria", expanded=True):
            st.markdown("### " + CRISIS_RESOURCES['title'])
            st.markdown(CRISIS_RESOURCES['message'])
            for resource in CRISIS_RESOURCES['resources']:
                st.markdown("- " + resource)
            st.markdown("*" + CRISIS_RESOURCES['hospital_note'] + "*")
            st.markdown("---")
            st.markdown("**You are not alone. Reaching out is a sign of strength.**")

    # Flagged sentences
    if suicidal_flagged:
        with st.expander("Sentences Flagged for Review",
                         expanded=risk_level in ['high', 'moderate_high']):
            st.caption(
                "The following sentences were identified by the model as containing "
                "indicators associated with suicidal ideation."
            )
            for sent in suicidal_flagged:
                st.markdown("- *" + html_lib.escape(sent['sentence']) + "*")

    # Nigerian wisdom
    with st.expander("Nigerian Wisdom", expanded=False):
        st.markdown("**Words of encouragement from our culture:**")
        for proverb in NIGERIAN_PROVERBS:
            st.markdown("*\"" + proverb + "\"*")

    st.markdown("---")
    render_progress_tracking()

    st.markdown("---")
    render_feedback_section()

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("New Check-In", use_container_width=True):
            for q in QUESTIONNAIRE:
                st.session_state["input_" + q['id']] = ""
            for key in ['responses', 'analyzer', 'results', 'analysis_complete']:
                st.session_state.pop(key, None)
            st.session_state.page = 'questionnaire'
            st.rerun()

    with col2:
        if st.button("Home", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()

    with col3:
        nat_resources   = NIGERIA_EMERGENCY_NUMBERS['national']['resources']
        state_resources = NIGERIA_EMERGENCY_NUMBERS['state_emergency']['resources']

        report_lines = [
            "HEEDX AI - MENTAL HEALTH CHECK-IN REPORT",
            "Generated  : " + datetime.now().strftime('%Y-%m-%d %H:%M'),
            "",
            "PRIMARY INDICATOR  : " + results['primary_indicator'].upper() +
            " (" + f"{results['primary_percentage']:.1f}" + "%)",
        ]
        if results['secondary_indicator']:
            report_lines.append(
                "SECONDARY INDICATOR: " + results['secondary_indicator'].upper() +
                " (" + f"{results['secondary_percentage']:.1f}" + "%)"
            )
        report_lines += [
            "",
            "RISK LEVEL: " + risk_level.upper(),
            "",
            "DISTRIBUTION:",
            *["  - " + k + ": " + f"{v:.1f}" + "%" for k, v in results['distribution'].items()],
            "",
            "RECOMMENDATIONS:",
            *["  " + str(i) + ". " + tip for i, tip in enumerate(recommendations['tips'], 1)],
            "",
            "NIGERIAN EMERGENCY RESOURCES:",
            *["  - " + r for r in nat_resources],
            *["  - " + r for r in state_resources],
            "",
            "DISCLAIMER: This is an AI-generated assessment, not a medical diagnosis.",
            "If you are in crisis, call 112 or visit the nearest Federal Neuropsychiatric Hospital.",
        ]

        st.download_button(
            label="Download Report",
            data="\n".join(report_lines),
            file_name="heedx_report_" + datetime.now().strftime('%Y%m%d_%H%M') + ".txt",
            mime="text/plain",
            use_container_width=True
        )


# ============================================
# PROGRESS PAGE
# ============================================
def render_progress_page():
    """Render the standalone progress page"""

    st.markdown("<h2 class='sub-header'>Your Wellness Journey</h2>", unsafe_allow_html=True)

    if not st.session_state.tracker:
        st.info("Complete your first check-in to see your progress.")
        if st.button("Start Check-In", use_container_width=True):
            st.session_state.page = 'questionnaire'
            st.rerun()
        return

    render_progress_tracking()

    if st.button("Back to Home", use_container_width=True):
        st.session_state.page = 'home'
        st.rerun()


# ============================================
# MAIN
# ============================================
def main():
    """Main application controller"""

    init_session_state()

    # Load model before sidebar renders — chatbot context depends on it
    if 'model' not in st.session_state:
        with st.spinner("Loading HeedX AI..."):
            st.session_state.model = load_model()

    model = st.session_state.model

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:0.5rem 0;'>"
            "<span style='font-size:1.5rem;font-weight:700;color:#008751;'>HeedX AI</span><br>"
            "<span style='font-size:0.78rem;color:#718096;'>Nigerian Mental Wellness Platform</span>"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("<div class='nigeria-accent'></div>", unsafe_allow_html=True)

        # Navigation
        st.markdown("**Navigation**")
        if st.button("Home", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
        if st.button("Check-In", use_container_width=True):
            st.session_state.page = 'questionnaire'
            st.rerun()
        if st.button("Wellness Chat", use_container_width=True):
            st.session_state.page = 'chat'
            st.rerun()
        if st.session_state.get('analysis_complete', False):
            if st.button("Summary", use_container_width=True):
                st.session_state.page = 'summary'
                st.rerun()
        if st.session_state.tracker and st.session_state.tracker.has_history():
            if st.button("Progress", use_container_width=True):
                st.session_state.page = 'progress'
                st.rerun()

        st.markdown("---")

        with st.expander("Emergency Numbers", expanded=False):
            st.markdown("**Emergency:** 112")
            for r in NIGERIA_EMERGENCY_NUMBERS['national']['resources'][:3]:
                st.markdown("- " + r)

        with st.expander("Disclaimer"):
            st.markdown(
                "This tool is for informational and supportive purposes only. "
                "It is not a substitute for professional medical advice. "
                "If you are in crisis, call **112** immediately."
            )

        # Contact channels — always visible
        render_sidebar_contacts()

    # ── Guard against missing model ───────────────────────────────────────────
    if model is None:
        st.error(
            "The model could not be loaded. Please ensure "
            "'psychotherapy_svm_model.pkl' is present in the application directory."
        )
        st.stop()

    # ── Page dispatch ─────────────────────────────────────────────────────────
    PAGE_MAP = {
        'home':          lambda: render_home_page(),
        'questionnaire': lambda: render_questionnaire_page(),
        'analysis':      lambda: render_analysis_page(model),
        'summary':       lambda: render_summary_page(),
        'progress':      lambda: render_progress_page(),
        'chat':          lambda: render_chat_page(),
    }

    page_fn = PAGE_MAP.get(st.session_state.page, lambda: render_home_page())
    page_fn()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div class='disclaimer'>
        <p><strong>HeedX AI &mdash; Nigerian Edition</strong></p>
        <p>This is a supportive tool only and does not constitute a medical diagnosis.</p>
        <p>Crisis lines: <strong>112</strong> (National Emergency) &nbsp;|&nbsp;
           <strong>08062106493</strong> (Mentally Aware Nigeria Initiative)</p>
        <p>You are not alone. Help is always available.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()