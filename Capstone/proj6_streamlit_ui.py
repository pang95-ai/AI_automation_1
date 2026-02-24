"""ABB8 2026 Slide Deck Chatbot — Hybrid RAG + Streamlit with MCP support.

Updated version that supports:
- Local PDFs or Google Drive links
- Flexible configuration via environment variables
- MCP server integration
"""

import hashlib
import logging
import os
from pathlib import Path

import openai
import streamlit as st
from dotenv import load_dotenv

from chatbot.config import ChatbotConfig
from chatbot.core import ChatbotDependencies, answer_question, get_index, make_retriever
from chatbot.pdf_handler import resolve_pdf_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_api_key(env_file: str) -> str:
    """Load and validate the OpenAI API key."""
    load_dotenv(env_file)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY not found in {env_file}")
    return api_key


@st.cache_resource(show_spinner=False)
def load_chatbot_resources(
    _config: ChatbotConfig, _pdf_hash: str
) -> tuple:
    """Build or load the index and wire up the retriever. Cached for the session."""
    api_key = load_api_key(_config.env_file)

    deps = ChatbotDependencies(
        chunk_size=_config.chunk_size,
        chunk_overlap=_config.chunk_overlap,
        top_k=_config.top_k,
        embed_model=_config.embed_model,
        llm_model=_config.llm_model,
        system_prompt=_config.system_prompt,
        llm_temperature=_config.llm_temperature,
        max_history=_config.max_history,
    )

    index, nodes = get_index(_config.pdf_path, api_key, _config.index_dir, deps)
    retriever = make_retriever(index, nodes, _config.top_k)
    llm_client = openai.OpenAI(api_key=api_key)
    return retriever, llm_client, nodes


def _render_sidebar(config: ChatbotConfig, slide_count: int) -> ChatbotConfig:
    """Render sidebar with status info and PDF source selection. Returns updated config."""
    with st.sidebar:
        st.title("ABB8 2026 Chatbot")
        st.markdown("---")

        # PDF Source Selection
        st.subheader("PDF Source")
        pdf_source_type = st.radio(
            "Choose PDF source:",
            ["Local File", "Google Drive Link"],
            key="pdf_source_type",
        )

        updated_pdf_path = config.pdf_path

        if pdf_source_type == "Google Drive Link":
            gdrive_url = st.text_input(
                "Paste Google Drive link:",
                placeholder="https://drive.google.com/file/d/...",
                key="gdrive_url",
            )
            if gdrive_url and st.button("Load From Google Drive"):
                try:
                    with st.spinner("Downloading PDF..."):
                        updated_pdf_path = resolve_pdf_path(gdrive_url)
                        st.success(f"Loaded: {updated_pdf_path.name}")
                except Exception as exc:
                    st.error(f"Failed to load PDF: {exc}")
        else:
            local_pdf = st.text_input(
                "Local PDF path:",
                value=str(config.pdf_path),
                key="local_pdf_path",
            )
            if local_pdf:
                updated_pdf_path = Path(local_pdf)

        config.pdf_path = updated_pdf_path

        st.markdown("---")
        st.success("Index loaded")
        st.markdown(f"**Slides indexed:** {slide_count}")
        st.markdown(f"**Embed model:** `{config.embed_model}`")
        st.markdown(f"**LLM:** `{config.llm_model}`")
        st.markdown(f"**Retrieval:** Hybrid (BM25 + Vector, top-{config.top_k})")
        st.markdown("---")

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        if st.button("Refresh Index"):
            st.cache_resource.clear()
            st.rerun()

    return config


def _render_message(role: str, content: str, slides: list[int] = None) -> None:
    """Render a single chat message with optional slide attribution."""
    if slides is None:
        slides = []

    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant" and slides:
            slide_refs = ", ".join(str(n) for n in slides)
            st.caption(f"Sources: Slides {slide_refs}")


def main() -> None:
    """Streamlit entry point: render the full chatbot UI."""
    st.set_page_config(
        page_title="ABB8 2026 Slide Chatbot",
        page_icon="🎓",
        layout="wide",
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "config" not in st.session_state:
        st.session_state.config = ChatbotConfig.load_from_env()

    # Load configuration
    config = st.session_state.config

    # Render sidebar and get updated config
    with st.spinner("Loading index (first run builds embeddings — may take ~2 min)..."):
        config = _render_sidebar(config, slide_count=sum(1 for _ in __import__("chatbot.core", fromlist=["extract_slides"]).extract_slides(config.pdf_path)))
        st.session_state.config = config

        # Create cache key based on PDF path to invalidate when PDF changes
        pdf_hash = hashlib.md5(str(config.pdf_path).encode()).hexdigest()
        retriever, llm_client, nodes = load_chatbot_resources(config, pdf_hash)

    # Main chat interface
    st.title("ABB8 2026 Slide Deck Chatbot")
    st.caption("Ask anything about the AI Builders Bootcamp 8 course material.")

    # Render message history
    for msg in st.session_state.messages:
        _render_message(msg["role"], msg["content"], msg.get("slides", []))

    # Chat input
    question = st.chat_input("Type your question...")
    if not question:
        return

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": question, "slides": []})
    _render_message("user", question, [])

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching slides..."):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]

            deps = ChatbotDependencies(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                top_k=config.top_k,
                embed_model=config.embed_model,
                llm_model=config.llm_model,
                system_prompt=config.system_prompt,
                llm_temperature=config.llm_temperature,
                max_history=config.max_history,
            )

            answer, slides = answer_question(retriever, question, history, llm_client, deps)

        st.markdown(answer)
        if slides:
            slide_refs = ", ".join(str(n) for n in slides)
            st.caption(f"Sources: Slides {slide_refs}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "slides": slides})


if __name__ == "__main__":
    main()
