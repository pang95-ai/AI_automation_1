"""ABB8 2026 Slide Deck Chatbot — Hybrid RAG + Streamlit.

Parses the combined slide PDF, builds a hybrid retrieval index
(dense vector search + BM25 keyword search), and serves a
conversational Q&A interface via Streamlit.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import openai
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PDF_PATH = Path("Capstone/2026_ABB8_Combined Slides.pdf")
INDEX_DIR = Path("Capstone/index_storage")
ENV_FILE = "OpenAI_key.env"

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 512        # approximate token limit per node
CHUNK_OVERLAP = 50      # token overlap when splitting long slides
TOP_K = 6               # nodes retrieved per hybrid query
LLM_TEMPERATURE = 0.2   # low for factual Q&A
MAX_HISTORY = 6         # conversation turns passed to the LLM

SYSTEM_PROMPT = (
    "You are an expert teaching assistant for the AI Builders Bootcamp 8 "
    "(ABB8) 2026. Answer questions using only the slide context provided. "
    "Be concise and direct. Cite slide numbers when relevant. "
    "If the context does not contain enough information, say so clearly. "
    "Do not invent facts not present in the slides."
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SlideChunk:
    """Extracted text from a single slide page."""

    page_num: int   # 1-indexed slide number
    title: str      # first non-empty line
    body: str       # remaining text lines joined


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_api_key() -> str:
    """Load and validate the OpenAI API key from the project env file."""
    load_dotenv(ENV_FILE)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY not found in {ENV_FILE}")
    return api_key


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------


def _parse_page_text(raw_text: str, page_num: int) -> SlideChunk | None:
    """Return a SlideChunk from a page's raw text, or None if the page is empty."""
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    if not lines:
        return None
    title = lines[0]
    body = "\n".join(lines[1:])
    return SlideChunk(page_num=page_num, title=title, body=body)


def extract_slides(pdf_path: Path) -> list[SlideChunk]:
    """Extract one SlideChunk per non-empty page from the PDF."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    slides: list[SlideChunk] = []
    with fitz.open(str(pdf_path)) as doc:
        for page_index, page in enumerate(doc, start=1):
            raw_text = page.get_text("text")
            chunk = _parse_page_text(raw_text, page_index)
            if chunk is not None:
                slides.append(chunk)

    logger.info("Extracted %d non-empty slides from %s", len(slides), pdf_path.name)
    return slides


# ---------------------------------------------------------------------------
# Node construction
# ---------------------------------------------------------------------------


def _slide_to_text(slide: SlideChunk) -> str:
    return f"Slide {slide.page_num}: {slide.title}\n{slide.body}"


def slides_to_nodes(slides: list[SlideChunk]) -> list[TextNode]:
    """Convert SlideChunks into LlamaIndex TextNodes with slide metadata.

    Long slides are sub-chunked with a SentenceSplitter while preserving
    the original slide_num and title in every resulting node's metadata.
    """
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes: list[TextNode] = []

    for slide in slides:
        full_text = _slide_to_text(slide)
        metadata = {"slide_num": slide.page_num, "title": slide.title}

        sub_texts = splitter.split_text(full_text)
        for sub_text in sub_texts:
            nodes.append(TextNode(text=sub_text, metadata=metadata))

    logger.info("Created %d TextNodes from %d slides", len(nodes), len(slides))
    return nodes


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------


def _configure_embedding(api_key: str) -> None:
    """Set the global LlamaIndex embedding model to OpenAI."""
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL, api_key=api_key)


def build_index(nodes: list[TextNode], api_key: str) -> VectorStoreIndex:
    """Embed all nodes, build a VectorStoreIndex, and persist it to disk."""
    _configure_embedding(api_key)
    logger.info("Building vector index from %d nodes...", len(nodes))
    index = VectorStoreIndex(nodes, show_progress=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))
    logger.info("Index persisted to %s", INDEX_DIR)
    return index


def load_index(api_key: str) -> VectorStoreIndex | None:
    """Load a previously persisted index from disk, or return None if absent."""
    if not INDEX_DIR.exists():
        return None
    _configure_embedding(api_key)
    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
        index = load_index_from_storage(storage_context)
        logger.info("Loaded existing index from %s", INDEX_DIR)
        return index
    except Exception as exc:
        logger.warning("Could not load index (%s); will rebuild.", exc)
        return None


def get_index(api_key: str) -> tuple[VectorStoreIndex, list[TextNode]]:
    """Return (index, nodes); loads from disk when available, else builds fresh."""
    slides = extract_slides(PDF_PATH)
    nodes = slides_to_nodes(slides)

    index = load_index(api_key)
    if index is None:
        index = build_index(nodes, api_key)

    return index, nodes


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------


def make_retriever(index: VectorStoreIndex, nodes: list[TextNode]) -> QueryFusionRetriever:
    """Combine dense vector retrieval and sparse BM25 via Reciprocal Rank Fusion."""
    vector_retriever = VectorIndexRetriever(index, similarity_top_k=TOP_K)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=TOP_K)
    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        mode="reciprocal_rerank",
        num_queries=1,
        similarity_top_k=TOP_K,
        use_async=False,
    )


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------


def _build_context(scored_nodes: list[NodeWithScore]) -> str:
    """Format retrieved nodes into a numbered context block for the LLM."""
    parts = []
    for i, node_with_score in enumerate(scored_nodes, start=1):
        slide_num = node_with_score.node.metadata.get("slide_num", "?")
        text = node_with_score.node.get_content()
        parts.append(f"[{i}] Slide {slide_num}:\n{text}")
    return "\n\n".join(parts)


def _extract_slide_nums(scored_nodes: list[NodeWithScore]) -> list[int]:
    """Return sorted unique slide numbers from a list of retrieved nodes."""
    nums = {node.node.metadata["slide_num"] for node in scored_nodes if "slide_num" in node.node.metadata}
    return sorted(nums)


def answer_question(
    retriever: QueryFusionRetriever,
    question: str,
    history: list[dict],
    llm_client: openai.OpenAI,
) -> tuple[str, list[int]]:
    """Retrieve relevant slides, then generate an answer using the LLM.

    Returns:
        A tuple of (answer_text, slide_numbers_used).
    """
    scored_nodes = retriever.retrieve(question)
    context = _build_context(scored_nodes)
    slide_nums = _extract_slide_nums(scored_nodes)

    trimmed_history = history[-(MAX_HISTORY * 2):]
    user_message = f"Context from slides:\n{context}\n\nQuestion: {question}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": user_message})

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
        )
    except openai.APIError as exc:
        logger.error("OpenAI API error: %s", exc)
        raise

    answer = response.choices[0].message.content or ""
    return answer, slide_nums


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_chatbot_resources() -> tuple[QueryFusionRetriever, openai.OpenAI]:
    """Build or load the index and wire up the retriever. Cached for the session."""
    api_key = load_api_key()
    index, nodes = get_index(api_key)
    retriever = make_retriever(index, nodes)
    llm_client = openai.OpenAI(api_key=api_key)
    return retriever, llm_client


def _render_sidebar(slide_count: int) -> None:
    """Render the sidebar with status info and a clear-chat button."""
    with st.sidebar:
        st.title("ABB8 2026 Chatbot")
        st.markdown("---")
        st.success("Index loaded")
        st.markdown(f"**Slides indexed:** {slide_count}")
        st.markdown(f"**Embed model:** `{EMBED_MODEL}`")
        st.markdown(f"**LLM:** `{LLM_MODEL}`")
        st.markdown(f"**Retrieval:** Hybrid (BM25 + Vector, top-{TOP_K})")
        st.markdown("---")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()


def _render_message(role: str, content: str, slides: list[int]) -> None:
    """Render a single chat message with optional slide attribution."""
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

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.spinner("Loading index (first run builds embeddings — may take ~2 min)..."):
        retriever, llm_client = load_chatbot_resources()

    slide_count = sum(1 for _ in extract_slides(PDF_PATH))
    _render_sidebar(slide_count)

    st.title("ABB8 2026 Slide Deck Chatbot")
    st.caption("Ask anything about the AI Builders Bootcamp 8 course material.")

    for msg in st.session_state.messages:
        _render_message(msg["role"], msg["content"], msg.get("slides", []))

    question = st.chat_input("Type your question...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question, "slides": []})
    _render_message("user", question, [])

    with st.chat_message("assistant"):
        with st.spinner("Searching slides..."):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            answer, slides = answer_question(retriever, question, history, llm_client)
        st.markdown(answer)
        if slides:
            slide_refs = ", ".join(str(n) for n in slides)
            st.caption(f"Sources: Slides {slide_refs}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "slides": slides})


if __name__ == "__main__":
    main()
