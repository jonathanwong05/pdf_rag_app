# app.py

import os
import streamlit as st
from pathlib import Path
from datetime import datetime
import shutil

# Import our refactored ingest/search functions
from src.ingest import ingest_pdfs
from src.search import search_embeddings, generate_rag_response
from src.utils import INDEX_NAME, DOC_PREFIX

# ─── 1. PAGE CONFIGURATION ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔍 Ask Any PDF (Local RAG)",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("📄 Ask Any PDF – Local RAG System")
st.markdown(
    """
    Upload your PDF documents, let the app index them, and then ask questions in plain English.
    The system will retrieve relevant text snippets and use an LLM (Vicuna, Mistral, Llama2, or OpenAI) to answer.
    """
)

# ─── 2. SIDEBAR SETTINGS ─────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

# 2.1. LLM Model Selection
model_name = st.sidebar.selectbox(
    "Choose LLM Model",
    options=[
        "vicuna:7b",
        "mistral:latest",
        "llama2:7b",
        # "gpt-3.5-turbo",  # Uncomment if you add OpenAI support
    ],
    index=0,
)

# 2.2. Prompt Template (optional advanced)
default_prompt = (
    "You are a helpful assistant. Use the following context excerpts from user‐uploaded PDFs to answer their question.\n"
    "If none of the context is relevant, say \"I don't know.\"\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{query}\n\n"
    "Answer:"
)

prompt_template = st.sidebar.text_area(
    "Prompt Template (advanced)",
    value=default_prompt,
    height=200,
)

st.sidebar.markdown("""---
*Note: Use `{context}` and `{query}` as placeholders.*""")

# 2.3. Ingestion Controls
st.sidebar.header("📥 Ingestion")
ingest_button = st.sidebar.button("🔄 Build/Refresh Index")

# ─── 3. MAIN AREA: PDF UPLOAD + INGESTION ────────────────────────────────────────
st.subheader("1️⃣ Upload PDF(s) to Index")

upload_col1, upload_col2 = st.columns([3, 1])

with upload_col1:
    uploaded_files = st.file_uploader(
        label="Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

with upload_col2:
    if st.button("📂 Clear All PDFs"):
        # Remove everything in ./data/ and clear Redis, forcing user to re-ingest
        if os.path.exists("data"):
            shutil.rmtree("data")
        os.makedirs("data", exist_ok=True)
        st.write("Removed all uploaded PDFs. Please upload new files.")

# Ensure the data folder exists
os.makedirs("data", exist_ok=True)

# Save any newly uploaded PDFs to disk
newly_uploaded = False
for pdf in uploaded_files or []:
    pdf_path = Path("data") / pdf.name
    if not pdf_path.exists():
        with open(pdf_path, "wb") as f:
            f.write(pdf.getbuffer())
        newly_uploaded = True

if newly_uploaded:
    st.success(f"Saved {len(uploaded_files)} file(s) to `./data/`. Click **Build/Refresh Index** to re‐ingest.")

# ─── 4. INGESTION: TRIGGER REDIS INDEX BUILD ───────────────────────────────────
if ingest_button:
    # Check if there are any PDFs in ./data/
    pdfs_in_folder = list(Path("data").glob("*.pdf"))
    if not pdfs_in_folder:
        st.warning("No PDFs found in `./data/`. Please upload at least one PDF.")
    else:
        with st.spinner("⏳ Ingesting PDFs and building Redis index…"):
            ingest_pdfs(data_dir="data/")
        st.success("✅ Ingestion complete! You can now ask questions below.")

# ─── 5. QUERY INTERFACE ─────────────────────────────────────────────────────────
st.subheader("2️⃣ Ask a Question")

query = st.text_input("Your question:", placeholder="E.g. “What is the definition of community‐acquired pneumonia?”")

# Only allow “Ask” if Redis index exists
redis_ready = False
try:
    import redis
    r = redis.Redis(host="localhost", port=6379, db=0)
    # Check if our index exists
    idx_info = r.ft(INDEX_NAME).info()  # If index doesn’t exist, will raise
    redis_ready = True
except Exception:
    redis_ready = False

if not redis_ready:
    st.info("🔴 No Redis index found. Upload PDFs and click **Build/Refresh Index** first.")
else:
    ask_button = st.button("❓ Ask")

    if ask_button and query.strip() != "":
        with st.spinner("🔍 Retrieving relevant context…"):
            context_results = search_embeddings(query, top_k=3)

        if not context_results:
            st.warning("⚠️ No relevant chunks found for your query.")
        else:
            # Display retrieved chunks
            st.markdown("**Top Retrieved Snippets:**")
            for i, res in enumerate(context_results, start=1):
                st.markdown(
                    f"**{i}.** From `{res['file']}` (page {res['page']}, similarity {res['similarity']:.2f}):\n> {res['chunk'][:300]}…"
                )

            # Now generate the RAG response
            with st.spinner("🤖 Generating LLM answer…"):
                answer = generate_rag_response(
                    query=query,
                    context_results=context_results,
                    model_name=model_name,
                    prompt_template=prompt_template,
                )
            st.markdown("**LLM Answer:**")
            st.write(answer)

# ─── 6. FOOTER / TIPS ───────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    """
    *Behind the scenes, this app uses Redis with an HNSW vector index to store PDF-chunk embeddings (384-dim,
    via MiniLM). When you “Ask” a question, it embeds your query, retrieves top-K chunks, then prompts an LLM
    (Vicuna, Mistral, Llama2, or—if configured—OpenAI) to synthesize a final answer.*
    """
)