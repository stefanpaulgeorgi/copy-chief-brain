"""
Copy Chief Brain - Streamlit App

A RAG-powered copy review tool that provides feedback in Stefan's voice.
"""

import streamlit as st
from pathlib import Path
import sys
import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime
import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.retriever import get_retriever
from src.generator import get_generator, get_chat_generator
from src import config

# Sessions directory
SESSIONS_DIR = Path(__file__).parent / "data" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Page config
st.set_page_config(
    page_title="Copy Chief Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stTextArea textarea {
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 14px;
    }
    .feedback-box {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
    }
    .chat-user {
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        margin-left: 20%;
        border-left: 3px solid #4a9eff;
        font-style: italic;
    }
    .chat-assistant {
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        margin-right: 5%;
        border-left: 3px solid #ff6b6b;
    }
    .session-card {
        background-color: #2d2d2d;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        cursor: pointer;
    }
    .session-card:hover {
        background-color: #3d3d3d;
    }
</style>
""", unsafe_allow_html=True)


def save_session(session_data: dict) -> str:
    """Save a session to disk and return the session ID."""
    session_id = session_data.get('id') or str(uuid.uuid4())[:8]
    session_data['id'] = session_id
    session_data['updated_at'] = datetime.now().isoformat()

    if 'created_at' not in session_data:
        session_data['created_at'] = session_data['updated_at']

    filepath = SESSIONS_DIR / f"{session_id}.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

    return session_id


def load_session(session_id: str) -> dict:
    """Load a session from disk."""
    filepath = SESSIONS_DIR / f"{session_id}.json"
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def list_sessions() -> list:
    """List all saved sessions, sorted by most recent."""
    sessions = []
    for filepath in SESSIONS_DIR.glob("*.json"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sessions.append({
                    'id': data.get('id', filepath.stem),
                    'copy_type': data.get('copy_type', 'Unknown'),
                    'copy_preview': data.get('copy_text', '')[:100] + '...' if len(data.get('copy_text', '')) > 100 else data.get('copy_text', ''),
                    'created_at': data.get('created_at', ''),
                    'updated_at': data.get('updated_at', ''),
                    'message_count': len(data.get('chat_messages', [])),
                })
        except:
            continue

    # Sort by updated_at descending
    sessions.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
    return sessions


def delete_session(session_id: str) -> bool:
    """Delete a session."""
    filepath = SESSIONS_DIR / f"{session_id}.json"
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from uploaded PDF file."""
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text_parts = []
        for page in pdf_reader.pages:
            text_parts.append(page.extract_text())
        return "\n\n".join(text_parts)
    except ImportError:
        st.error("PyPDF2 not installed. Run: pip3 install PyPDF2")
        return ""
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


def extract_text_from_url(url: str) -> str:
    """Fetch and extract text content from a URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()

        main_content = soup.find('main') or soup.find('article') or soup.find('body')

        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text
    except requests.RequestException as e:
        st.error(f"Error fetching URL: {e}")
        return ""
    except Exception as e:
        st.error(f"Error processing URL: {e}")
        return ""


def init_session_state():
    """Initialize session state variables."""
    if 'feedback_history' not in st.session_state:
        st.session_state.feedback_history = []
    if 'current_feedback' not in st.session_state:
        st.session_state.current_feedback = None
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'current_copy' not in st.session_state:
        st.session_state.current_copy = ""
    if 'review_complete' not in st.session_state:
        st.session_state.review_complete = False
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    if 'current_copy_type' not in st.session_state:
        st.session_state.current_copy_type = "General/Other"
    if 'current_user_context' not in st.session_state:
        st.session_state.current_user_context = ""
    if 'current_awareness_level' not in st.session_state:
        st.session_state.current_awareness_level = "Not Sure / Mixed"


def auto_save_session():
    """Auto-save the current session if there's content."""
    if st.session_state.current_feedback and st.session_state.current_copy:
        session_data = {
            'id': st.session_state.current_session_id,
            'copy_text': st.session_state.current_copy,
            'copy_type': st.session_state.current_copy_type,
            'awareness_level': st.session_state.current_awareness_level,
            'user_context': st.session_state.current_user_context,
            'feedback': st.session_state.current_feedback,
            'chat_messages': st.session_state.chat_messages,
        }
        session_id = save_session(session_data)
        st.session_state.current_session_id = session_id


# Copy type descriptions for the AI
COPY_TYPE_CONTEXT = {
    "Product Detail Page (PDP)": "This is a Product Detail Page (PDP) for e-commerce. Long-form copy is often appropriate here as buyers research before purchasing. Don't critique length unless it's truly excessive or unfocused. Focus on: persuasion, benefit clarity, objection handling, and conversion elements.",
    "Landing Page": "This is a landing page designed to convert visitors on a specific offer. Evaluate: headline strength, benefit clarity, social proof, urgency, and CTA effectiveness.",
    "VSL Script": "This is a Video Sales Letter script. Evaluate: hook strength, story flow, mechanism explanation, offer presentation, and emotional arc. Length varies by offer complexity.",
    "Advertorial": "This is an advertorial (ad disguised as editorial content). Evaluate: native feel, story engagement, transition to pitch, and credibility elements.",
    "Email": "This is an email. Evaluate: subject line potential, hook, readability, CTA clarity. Keep feedback focused on email-specific best practices.",
    "Ad (Short-form)": "This is a short-form ad (Facebook, YouTube, etc). Evaluate: hook punch, curiosity creation, CTA. Brevity is expected.",
    "Hook/Lead Only": "This is just a hook or lead section. Focus feedback entirely on: attention-grabbing power, curiosity creation, and transition potential.",
    "Sales Page": "This is a sales page. Evaluate: headline, lead, story, mechanism, offer, guarantee, and CTA stack. Longer copy is normal here.",
    "Upsell Page": "This is an upsell/OTO page. Evaluate: continuity from main offer, value prop, urgency, and whether it enhances without undermining the initial purchase.",
    "General/Other": "Review this copy using general direct response principles. Ask clarifying questions if the format is unclear."
}

COPY_TYPE_OPTIONS = list(COPY_TYPE_CONTEXT.keys())

# Awareness level descriptions (Eugene Schwartz's 5 Levels)
AWARENESS_LEVEL_CONTEXT = {
    "1 - Unaware": "The prospect doesn't know they have a problem. Copy should lead with story/emotion and reveal the problem. Don't mention product early. Focus on pattern interrupt and problem revelation.",
    "2 - Problem Aware": "The prospect knows they have a problem but doesn't know solutions exist. Copy should agitate the pain, then introduce the solution category. Lead with emotional entry point, not mechanism.",
    "3 - Solution Aware": "The prospect knows solutions exist but doesn't know your specific product. Copy should lead with your unique mechanism and differentiation. Why is YOUR solution different/better?",
    "4 - Product Aware": "The prospect knows your product but isn't convinced yet. Copy should lead with proof, testimonials, and objection handling. Address skepticism directly.",
    "5 - Most Aware": "The prospect knows everything and just needs the deal. Copy should lead with offer, urgency, and price. Get to the point fast.",
    "Not Sure / Mixed": "Awareness level is unclear or the copy needs to work across multiple awareness levels. Review will note where awareness-level adjustments might help."
}

AWARENESS_LEVEL_OPTIONS = list(AWARENESS_LEVEL_CONTEXT.keys())


def main():
    init_session_state()

    # Header
    st.title("🧠 Copy Chief Brain")
    st.markdown("*Get instant copy feedback in Stefan's voice*")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        retriever = get_retriever()
        stats = retriever.get_stats()

        if stats['status'] == 'ready':
            st.success(f"✅ RAG Ready: {stats['count']:,} chunks indexed")
        else:
            st.warning("⚠️ No transcripts indexed yet")

        st.divider()

        st.subheader("Review Options")

        use_rag = st.checkbox(
            "Use RAG context",
            value=True,
            help="When ON, retrieves relevant examples from Stefan's past reviews."
        )

        top_k = st.slider(
            "Context chunks",
            min_value=3,
            max_value=15,
            value=8,
            help="How many examples from past reviews to include."
        )

        st.divider()

        # === SAVED SESSIONS ===
        st.subheader("📚 Saved Sessions")

        sessions = list_sessions()

        if sessions:
            for session in sessions[:10]:  # Show last 10
                # Format date nicely
                try:
                    dt = datetime.fromisoformat(session['updated_at'])
                    date_str = dt.strftime("%b %d, %I:%M %p")
                except:
                    date_str = "Unknown"

                col_load, col_del = st.columns([4, 1])

                with col_load:
                    label = f"**{session['copy_type'][:15]}**\n{session['copy_preview'][:40]}...\n_{date_str}_"
                    if st.button(f"📄 {session['copy_type'][:20]}", key=f"load_{session['id']}", use_container_width=True):
                        # Load the session
                        loaded = load_session(session['id'])
                        if loaded:
                            st.session_state.current_session_id = loaded['id']
                            st.session_state.current_copy = loaded.get('copy_text', '')
                            st.session_state.current_copy_type = loaded.get('copy_type', 'General/Other')
                            st.session_state.current_awareness_level = loaded.get('awareness_level', 'Not Sure / Mixed')
                            st.session_state.current_user_context = loaded.get('user_context', '')
                            st.session_state.current_feedback = loaded.get('feedback', '')
                            st.session_state.chat_messages = loaded.get('chat_messages', [])
                            st.session_state.review_complete = bool(loaded.get('feedback'))
                            st.session_state.extracted_text = loaded.get('copy_text', '')
                            st.rerun()

                with col_del:
                    if st.button("🗑️", key=f"del_{session['id']}"):
                        delete_session(session['id'])
                        st.rerun()

                st.caption(f"{date_str} • {session['message_count']} msgs")

            if len(sessions) > 10:
                st.caption(f"... and {len(sessions) - 10} more")
        else:
            st.caption("No saved sessions yet")

        st.divider()

        st.subheader("ℹ️ How it works")
        st.markdown("""
        1. Submit copy + specify the type
        2. Add any context
        3. Get feedback, then chat to refine

        Sessions auto-save as you work.
        """)

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Submit Copy for Review")

        # Show current session ID if exists
        if st.session_state.current_session_id:
            st.caption(f"Session: {st.session_state.current_session_id}")

        # === COPY TYPE SELECTION ===
        copy_type_index = COPY_TYPE_OPTIONS.index(st.session_state.current_copy_type) if st.session_state.current_copy_type in COPY_TYPE_OPTIONS else 9

        copy_type = st.selectbox(
            "What type of copy is this? *",
            options=COPY_TYPE_OPTIONS,
            index=copy_type_index,
            help="This helps calibrate feedback. A 3000-word PDP is fine; a 3000-word ad is not."
        )
        st.session_state.current_copy_type = copy_type

        # === AWARENESS LEVEL SELECTION ===
        awareness_index = AWARENESS_LEVEL_OPTIONS.index(st.session_state.get('current_awareness_level', 'Not Sure / Mixed')) if st.session_state.get('current_awareness_level') in AWARENESS_LEVEL_OPTIONS else 5

        awareness_level = st.selectbox(
            "Target awareness level (Schwartz)",
            options=AWARENESS_LEVEL_OPTIONS,
            index=awareness_index,
            help="Eugene Schwartz's 5 levels: Unaware → Problem Aware → Solution Aware → Product Aware → Most Aware"
        )
        st.session_state.current_awareness_level = awareness_level

        # === CONTEXT FIELD ===
        user_context = st.text_area(
            "Context for the reviewer (optional but recommended)",
            value=st.session_state.current_user_context,
            height=120,
            placeholder="""Examples:
• "This is for a $97 supplement targeting men 45+. Length is intentional."
• "Main concern is whether the hook is strong enough."
• "This already converts at 2%, looking to beat control."
""",
            help="Anything you'd tell a human reviewer before they look at the copy"
        )
        st.session_state.current_user_context = user_context

        st.divider()

        # === INPUT METHODS ===
        input_tab1, input_tab2, input_tab3 = st.tabs(["✏️ Paste", "📄 Upload", "🔗 URL"])

        copy_text = st.session_state.current_copy or ""

        with input_tab1:
            copy_text_paste = st.text_area(
                "Paste your copy here",
                value=copy_text if not st.session_state.extracted_text else "",
                height=250,
                placeholder="Paste your copy here...",
                key="copy_input_paste"
            )
            if copy_text_paste:
                copy_text = copy_text_paste

        with input_tab2:
            uploaded_file = st.file_uploader(
                "Upload a file",
                type=['pdf', 'txt', 'md'],
                help="Supported: PDF, TXT, Markdown"
            )

            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    extracted = extract_text_from_pdf(uploaded_file)
                else:
                    extracted = uploaded_file.read().decode('utf-8', errors='ignore')

                if extracted:
                    st.session_state.extracted_text = extracted
                    st.success(f"✅ Extracted {len(extracted):,} characters")
                    with st.expander("Preview extracted text"):
                        st.text(extracted[:2000] + "..." if len(extracted) > 2000 else extracted)
                    copy_text = extracted

        with input_tab3:
            url_input = st.text_input(
                "Enter URL to review",
                placeholder="https://example.com/sales-page",
                key="url_input"
            )

            fetch_btn = st.button("🔍 Fetch Page", disabled=not url_input)

            if fetch_btn and url_input:
                with st.spinner("Fetching page content..."):
                    extracted = extract_text_from_url(url_input)
                    if extracted:
                        st.session_state.extracted_text = extracted
                        st.success(f"✅ Extracted {len(extracted):,} characters")
                        with st.expander("Preview extracted text"):
                            st.text(extracted[:2000] + "..." if len(extracted) > 2000 else extracted)
                        copy_text = extracted

            elif st.session_state.extracted_text and not copy_text:
                copy_text = st.session_state.extracted_text

        # === ACTION BUTTONS ===
        st.divider()

        col_btn1, col_btn2 = st.columns([1, 1])

        with col_btn1:
            submit_btn = st.button(
                "🔍 Get Feedback",
                type="primary",
                use_container_width=True,
                disabled=not copy_text.strip() if copy_text else True
            )

        with col_btn2:
            clear_btn = st.button(
                "🗑️ Start New Review",
                use_container_width=True
            )

        if clear_btn:
            st.session_state.current_feedback = None
            st.session_state.extracted_text = ""
            st.session_state.chat_messages = []
            st.session_state.current_copy = ""
            st.session_state.review_complete = False
            st.session_state.current_session_id = None
            st.session_state.current_copy_type = "General/Other"
            st.session_state.current_awareness_level = "Not Sure / Mixed"
            st.session_state.current_user_context = ""
            st.rerun()

        if copy_text:
            word_count = len(copy_text.split())
            st.caption(f"📊 {word_count:,} words | {len(copy_text):,} characters")

    with col2:
        st.subheader("💬 Stefan's Feedback")

        # === INITIAL REVIEW ===
        if submit_btn and copy_text and copy_text.strip():
            st.session_state.current_copy = copy_text
            st.session_state.chat_messages = []
            st.session_state.current_session_id = None  # New session

            copy_type_context = COPY_TYPE_CONTEXT.get(copy_type, "")
            awareness_context = AWARENESS_LEVEL_CONTEXT.get(awareness_level, "")
            full_context = f"**COPY TYPE:** {copy_type}\n{copy_type_context}"
            full_context += f"\n\n**TARGET AWARENESS LEVEL:** {awareness_level}\n{awareness_context}"
            if user_context.strip():
                full_context += f"\n\n**REVIEWER CONTEXT FROM USER:** {user_context}"

            with st.spinner("Stefan is reviewing your copy..."):
                try:
                    generator = get_generator()
                    feedback_placeholder = st.empty()
                    full_response = ""

                    # Calculate actual word count to pass to generator
                    actual_word_count = len(copy_text.split())

                    for chunk in generator.generate_feedback_streaming(
                        copy_text=copy_text,
                        copy_type=copy_type,
                        additional_context=full_context,
                        use_rag=use_rag,
                        top_k=top_k,
                        word_count=actual_word_count
                    ):
                        full_response += chunk
                        feedback_placeholder.markdown(full_response)

                    st.session_state.current_feedback = full_response
                    st.session_state.review_complete = True

                    st.session_state.chat_messages = [
                        {"role": "assistant", "content": full_response}
                    ]

                    # Auto-save
                    auto_save_session()

                except Exception as e:
                    st.error(f"Error generating feedback: {str(e)}")

        elif st.session_state.current_feedback:
            st.markdown(st.session_state.current_feedback)

            # === EXPORT BUTTONS ===
            col_copy1, col_copy2 = st.columns([1, 1])
            with col_copy1:
                # Combine all content for full export
                full_export = f"# Copy Chief Feedback\n\n"
                full_export += f"**Copy Type:** {st.session_state.current_copy_type}\n\n"
                full_export += f"**Awareness Level:** {st.session_state.current_awareness_level}\n\n"
                if st.session_state.current_user_context:
                    full_export += f"**Context:** {st.session_state.current_user_context}\n\n"
                full_export += f"---\n\n## Initial Feedback\n\n{st.session_state.current_feedback}\n\n"

                if len(st.session_state.chat_messages) > 1:
                    full_export += "---\n\n## Follow-up Conversation\n\n"
                    for msg in st.session_state.chat_messages[1:]:
                        if msg["role"] == "user":
                            full_export += f"**You:** {msg['content']}\n\n"
                        else:
                            full_export += f"**Stefan:** {msg['content']}\n\n"

                st.download_button(
                    "📥 Export Full Session",
                    data=full_export,
                    file_name=f"copy_chief_session_{st.session_state.current_session_id or 'new'}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

            with col_copy2:
                if st.button("📄 Copy to Clipboard", use_container_width=True, key="copy_main"):
                    st.code(st.session_state.current_feedback, language=None)
                    st.info("👆 Select all and copy (Cmd+A, Cmd+C)")

        else:
            st.info("👈 Add your copy, select the type, add any context, then click 'Get Feedback'")

        # === CHAT FOR FOLLOW-UP ===
        if st.session_state.review_complete:
            st.divider()
            st.markdown("### 💬 Continue the conversation")
            st.caption("Ask for rewrites, clarification, or dig deeper on specific points")

            # Display existing chat history
            if len(st.session_state.chat_messages) > 1:
                for i, msg in enumerate(st.session_state.chat_messages[1:]):
                    if msg["role"] == "user":
                        st.markdown(f'<div class="chat-user"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-assistant"><strong>Stefan:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)

                        st.download_button(
                            "📋 Download this response",
                            data=msg["content"],
                            file_name=f"stefan_response_{i}.md",
                            mime="text/markdown",
                            key=f"download_{i}_{hash(msg['content'][:20])}"
                        )

            # Chat input
            st.markdown("---")
            chat_input = st.text_area(
                "Your message",
                height=100,
                placeholder="""Examples:
• "Can you rewrite the hook with your suggestions?"
• "What would a stronger CTA look like?"
• "Rewrite the first 3 paragraphs incorporating your feedback"
""",
                key="chat_input"
            )

            if st.button("Send", type="primary", disabled=not chat_input.strip() if chat_input else True):
                if chat_input and chat_input.strip():
                    st.session_state.chat_messages.append({
                        "role": "user",
                        "content": chat_input
                    })

                    with st.spinner("Stefan is responding..."):
                        try:
                            chat_gen = get_chat_generator()
                            response = chat_gen.chat_response(
                                messages=st.session_state.chat_messages,
                                original_copy=st.session_state.current_copy,
                                use_rag=use_rag,
                                top_k=top_k
                            )

                            st.session_state.chat_messages.append({
                                "role": "assistant",
                                "content": response
                            })

                            # Auto-save after each chat
                            auto_save_session()

                            st.rerun()

                        except Exception as e:
                            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
