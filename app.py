"""
Document Q&A Chatbot (RAG Architecture)
Optimized for LinkedIn Showcase & Production Demonstration
Built with Streamlit, FAISS Vector Database, and LLM Grounded Retrieval
"""

import os
import time
import streamlit as st
from rag_engine import RAGEngine

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="DocuMind RAG | Document Q&A AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling (Dark Mode, Glassmorphism, Neon Accents, Clean Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Brand Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 27, 75, 0.9) 50%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.2);
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.2));
        border: 1px solid rgba(129, 140, 248, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        color: #818cf8;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.98rem;
        margin: 0;
        max-width: 860px;
        line-height: 1.5;
    }
    
    /* Pipeline Step Tracker */
    .pipeline-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 16px;
        padding-top: 14px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .pipeline-step {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 4px 10px;
        border-radius: 8px;
        color: #e2e8f0;
        font-size: 0.76rem;
        font-weight: 500;
    }
    
    .pipeline-step span.step-num {
        color: #a855f7;
        font-weight: 700;
    }
    
    /* Status Notification Banner */
    .status-banner {
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
    }
    
    .status-banner.connected {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-banner.offline {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 12px;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 16px;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(129, 140, 248, 0.4);
        transform: translateY(-2px);
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.74rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    
    .metric-val {
        color: #f8fafc;
        font-size: 1.45rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-sub {
        color: #10b981;
        font-size: 0.72rem;
        margin-top: 2px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    /* Source Chunk Cards */
    .source-box {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        padding: 12px 14px;
        margin-top: 8px;
        font-size: 0.86rem;
        color: #cbd5e1;
    }
    
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
        font-size: 0.76rem;
        color: #818cf8;
        font-weight: 600;
    }
    
    .badge-page {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 2px 8px;
        border-radius: 6px;
        color: #a5b4fc;
    }
    
    .badge-score {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 2px 8px;
        border-radius: 6px;
        color: #34d399;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False

if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}

if "sample_loaded" not in st.session_state:
    st.session_state.sample_loaded = False

if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""

if "active_query" not in st.session_state:
    st.session_state.active_query = None


# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    # Provider Selection
    provider_options = [
        "Google Gemini (Free & Recommended)",
        "Groq (Free Llama-3.3 70B)",
        "OpenAI (GPT-4o / Mini)",
        "Local Semantic Mode (Offline / Zero-Key)"
    ]
    provider_sel = st.selectbox(
        "AI Engine / Provider",
        options=provider_options,
        index=0,
        help="Select LLM provider. Google Gemini and Groq offer 100% free API tiers without requiring a credit card."
    )
    
    if "Gemini" in provider_sel:
        provider_key = "gemini"
    elif "Groq" in provider_sel:
        provider_key = "groq"
    elif "OpenAI" in provider_sel:
        provider_key = "openai"
    else:
        provider_key = "local"
    
    # API Key Input
    if provider_key in ["gemini", "openai", "groq"]:
        provider_label = "Gemini" if provider_key == "gemini" else ("Groq" if provider_key == "groq" else "OpenAI")
        key_input = st.text_input(
            f"{provider_label} API Key",
            value=st.session_state.api_key,
            type="password",
            placeholder="Paste your API key here...",
            help="Your key is stored only in this session and never logged."
        )
        if key_input != st.session_state.api_key:
            st.session_state.api_key = key_input
            if st.session_state.rag_engine:
                st.session_state.rag_engine.provider = provider_key
                st.session_state.rag_engine.api_key = key_input
                st.session_state.rag_engine._init_client()
            st.rerun()

        if st.session_state.api_key:
            st.success(f"🟢 {provider_label} API Connected", icon="✅")
        else:
            if provider_key == "gemini":
                st.markdown("[👉 **Get Free Gemini API Key**](https://aistudio.google.com/app/apikey) *(Free tier, zero credit card)*")
            elif provider_key == "groq":
                st.markdown("[👉 **Get Free Groq API Key**](https://console.groq.com/keys) *(Instant free signup)*")
    else:
        st.success("⚡ Local FAISS Engine Active (Offline / Zero-Key)")

    st.markdown("---")
    st.markdown("### 🛠️ RAG Hyperparameters")
    chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=1200, value=500, step=50,
                           help="Size of each document segment. Smaller chunks yield higher semantic precision.")
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=20, max_value=200, value=75, step=10,
                             help="Characters shared between adjacent chunks to preserve contextual continuity.")
    top_k = st.slider("Top-K Retrieved Chunks", min_value=2, max_value=8, value=4, step=1,
                      help="Number of nearest-neighbor chunks retrieved from FAISS per query.")

    st.markdown("---")
    st.markdown("### 🎯 Grounding Mode")
    strict_grounding = st.toggle("Enforce Strict Context Grounding", value=True,
                                help="When enabled, instructs the LLM to only respond using provided context, preventing hallucinations.")

    st.markdown("---")
    st.markdown("### 💼 LinkedIn Visibility Mode")
    show_linkedin_tools = st.checkbox("Show LinkedIn Showcase Toolkit", value=True,
                                      help="Displays copy-ready LinkedIn post templates, metrics, and architecture diagrams.")
    
    st.caption("🚀 Developed for LinkedIn Portfolio Showcase")


# Main Page Header & Showcase Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">⚡ PRODUCTION-READY RAG PIPELINE</div>
    <h1 class="hero-title">Document Q&A Chatbot (RAG)</h1>
    <p class="hero-subtitle">
        Upload any PDF and ask natural language questions. The app parses the document, splits text into semantic chunks,
        embeds them into dense vector space, indexes them via <b>FAISS Vector Database</b>, and retrieves the most relevant passages
        so the LLM answers strictly from your verified content.
    </p>
    <div class="pipeline-strip">
        <div class="pipeline-step"><span class="step-num">1</span> 📄 Load & Split (PDF Extraction)</div>
        <div class="pipeline-step"><span class="step-num">2</span> 🧠 Dense Vector Embeddings</div>
        <div class="pipeline-step"><span class="step-num">3</span> ⚡ FAISS Vector Store Index</div>
        <div class="pipeline-step"><span class="step-num">4</span> 🎯 Cosine Similarity Retrieval</div>
        <div class="pipeline-step"><span class="step-num">5</span> 🤖 Grounded Context Synthesis</div>
    </div>
</div>
""", unsafe_allow_html=True)


# Document Upload & Sample Loader Section
col_upload, col_sample = st.columns([2.5, 1.2])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload any PDF Document to Index",
        type=["pdf"],
        help="Upload any PDF (research paper, textbook, manual, lecture notes, resume, etc.)"
    )

with col_sample:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("⚡ Load Demo PDF (AI Agents Whitepaper)", use_container_width=True):
        sample_path = "sample_document.pdf"
        if not os.path.exists(sample_path):
            from create_sample_pdf import generate_sample_pdf
            generate_sample_pdf(sample_path)
            
        with open(sample_path, "rb") as f:
            uploaded_file = f.read()
        st.session_state.sample_loaded = True
        st.session_state.sample_bytes = uploaded_file
        st.rerun()

# Handle active document
active_file = uploaded_file or (st.session_state.sample_bytes if st.session_state.get("sample_loaded") else None)
active_filename = "sample_document.pdf (AI Whitepaper)" if st.session_state.get("sample_loaded") and not uploaded_file else (uploaded_file.name if uploaded_file else None)

# Trigger Indexing Process
if active_file is not None and (not st.session_state.doc_loaded or st.session_state.get("last_processed_file") != active_filename):
    with st.spinner("🔄 Indexing document: Segmenting into semantic chunks & building FAISS vector database..."):
        try:
            start_proc = time.perf_counter()
            engine = RAGEngine(
                provider=provider_key,
                api_key=st.session_state.api_key,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # 1. Load and split
            chunks = engine.load_and_split_pdf(active_file)
            
            # 2 & 3. Embed & Store in FAISS
            vector_count = engine.build_vector_store()
            indexing_time = (time.perf_counter() - start_proc) * 1000

            st.session_state.rag_engine = engine
            st.session_state.doc_loaded = True
            st.session_state.last_processed_file = active_filename
            
            total_chars = sum(c["char_count"] for c in chunks)
            total_words = sum(c.get("word_count", 0) for c in chunks)
            
            st.session_state.doc_stats = {
                "filename": active_filename,
                "pages": len(engine.raw_pages),
                "chunks": len(chunks),
                "total_words": total_words,
                "total_chars": total_chars,
                "dimension": engine.embedding_dimension,
                "index_latency": indexing_time,
                "suggested_questions": engine.generate_suggested_questions(active_filename)
            }
            # Clear previous chat messages on new document upload
            st.session_state.messages = []
            st.toast(f"✅ Indexed {len(chunks)} chunks into FAISS vector database in {indexing_time:.1f}ms!", icon="⚡")
        except Exception as e:
            st.error(f"❌ Error indexing document: {e}")


# Display Document Metrics Dashboard if loaded
if st.session_state.doc_loaded and st.session_state.doc_stats:
    stats = st.session_state.doc_stats
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Document Pages</div>
            <div class="metric-val">{stats['pages']}</div>
            <div class="metric-sub">📄 {stats['filename'][:22]}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Chunks</div>
            <div class="metric-val">{stats['chunks']}</div>
            <div class="metric-sub">✂️ ~{chunk_size} chars / chunk</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">FAISS Vector Space</div>
            <div class="metric-val">{stats['dimension']}D</div>
            <div class="metric-sub">📐 Normalized Cosine (IP)</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Index Build Time</div>
            <div class="metric-val">{stats['index_latency']:.1f}<span style="font-size:0.9rem">ms</span></div>
            <div class="metric-sub">⚡ Real-time FAISS index</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Grounding Mode</div>
            <div class="metric-val">100%</div>
            <div class="metric-sub">🎯 Strict Citation Required</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# AI Connection Status / Quick Key Connector
if st.session_state.api_key:
    provider_name = "Google Gemini" if provider_key == "gemini" else ("Groq" if provider_key == "groq" else "OpenAI")
    st.markdown(f"""
    <div class="status-banner connected">
        <div>
            <strong style="color: #10b981;">🟢 AI Connected: {provider_name}</strong>
            <span style="color: #94a3b8; font-size: 0.85rem; margin-left: 8px;">Full generative conversational synthesis active with FAISS grounding.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Offer 1-click Free Key guide right on the page
    with st.container():
        st.markdown("""
        <div class="status-banner offline">
            <div>
                <strong style="color: #818cf8;">⚡ Mode: Local FAISS Vector Search (No Key Required)</strong>
                <div style="color: #cbd5e1; font-size: 0.86rem; margin-top: 3px;">
                    The bot retrieves the exact matching excerpts from your PDF. 
                    <b>Want full conversational essays and detailed explanations?</b> Add a 100% free Google Gemini key below (takes 10s):
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        c_key, c_link = st.columns([3, 1.2])
        with c_key:
            inline_key = st.text_input("Paste Free Gemini API Key here:", type="password", key="inline_key_input",
                                       placeholder="AIzaSy... (Paste and press Enter to activate conversational AI)")
            if inline_key and inline_key != st.session_state.api_key:
                st.session_state.api_key = inline_key
                if st.session_state.rag_engine:
                    st.session_state.rag_engine.provider = "gemini"
                    st.session_state.rag_engine.api_key = inline_key
                    st.session_state.rag_engine._init_client()
                st.rerun()
        with c_link:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            st.markdown("[👉 **Get Free Gemini Key**](https://aistudio.google.com/app/apikey)")


# LinkedIn Showcase Toolkit (Templates & Architecture Specs)
if show_linkedin_tools:
    with st.expander("💼 LinkedIn Showcase Toolkit & Post Copy Generator", expanded=False):
        st.markdown("#### 🚀 Ready-to-Post LinkedIn Caption")
        doc_name_show = st.session_state.doc_stats.get('filename', 'Technical Document') if st.session_state.doc_loaded else 'Engineering Whitepaper'
        
        linkedin_text = f"""🚀 Built & Deployed an End-to-End Document Q&A Chatbot using RAG (Retrieval-Augmented Generation) & FAISS Vector Database! 🤖📄

Traditional LLMs alone don't know your private enterprise documents or university textbooks ({doc_name_show}). When asked about them, they either hallucinate or guess.

To solve this with 100% verifiable accuracy, I built a production-grade RAG pipeline from scratch.

🔍 How it works under the hood:
1️⃣ Load & Split: Ingested PDF text page-by-page and segmented it into semantic chunks ({chunk_size} chars with {chunk_overlap} chars overlap to preserve context continuity).
2️⃣ Dense Embeddings: Converted each chunk into high-dimensional vector representations.
3️⃣ FAISS Vector Store: Stored normalized embeddings in Facebook AI Similarity Search (FAISS IndexFlatIP) for sub-10ms Cosine Similarity search.
4️⃣ Top-K Semantic Retrieval: For every question, retrieved the {top_k} most contextually relevant chunks with similarity scoring.
5️⃣ Grounded LLM Synthesis: Injected retrieved chunks with strict prompt engineering ("Answer only from this context"), citing page numbers and preventing hallucinations.

💡 Key Highlights:
⚡ Sub-millisecond vector retrieval with FAISS IndexFlatIP
🎯 100% strictly grounded answers with page-level citations
🛠️ Built with Python, Streamlit, FAISS, PyPDF, and Gemini/OpenAI

Check out the demo video below! What are your favorite approaches to chunking and vector indexing? Let me know in the comments! 👇

#ArtificialIntelligence #RAG #MachineLearning #Python #FAISS #Streamlit #LLM #GenerativeAI #OpenAI #Gemini #DataEngineering"""

        st.code(linkedin_text, language="markdown")
        st.info("💡 Tip: Use screen recording software (or Windows Win+Alt+R) to record a 30-second interaction asking questions and showing the retrieved chunk citations!")


# Dynamic Document-Aware Suggested Questions
if st.session_state.doc_loaded:
    suggested_q = st.session_state.doc_stats.get("suggested_questions", [
        "What is the main topic covered in this document?",
        "Explain the key technical concepts described in the text.",
        "Summarize the main procedures or specifications."
    ])
    
    st.markdown("##### 💡 Suggested Questions for this Document (Click to Ask):")
    q_cols = st.columns(len(suggested_q))
    
    for i, col in enumerate(q_cols):
        with col:
            if st.button(f"👉 {suggested_q[i]}", key=f"sug_btn_{i}", use_container_width=True):
                st.session_state.active_query = suggested_q[i]
                st.rerun()
else:
    st.info("👋 Upload a PDF above or click **'⚡ Load Demo PDF'** to begin asking questions!", icon="👆")


# Chat History Display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"🔍 View {len(msg['sources'])} Retrieved FAISS Chunks & Page Citations"):
                for src in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-box">
                        <div class="source-header">
                            <span class="badge-page">Page {src.get('page', '?')} • Chunk #{src.get('id', '?')}</span>
                            <span class="badge-score">Similarity: {src.get('similarity_score', 0):.2%}</span>
                        </div>
                        <div>{src.get('text', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)


# Query Processing (either from text input or suggested question click)
user_query = st.chat_input("Ask any question about the document...")
if not user_query and st.session_state.active_query:
    user_query = st.session_state.active_query
    st.session_state.active_query = None

if user_query:
    if not st.session_state.doc_loaded or st.session_state.rag_engine is None:
        st.warning("⚠️ Please upload a PDF or click 'Load Demo PDF' first so the vector database can be indexed.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response
        with st.chat_message("assistant"):
            engine: RAGEngine = st.session_state.rag_engine
            
            with st.spinner("🔍 Querying FAISS vector store & retrieving top chunks..."):
                retrieved_chunks = engine.retrieve(user_query, top_k=top_k)
                latency = engine.last_search_latency_ms

            with st.spinner("🤖 Synthesizing grounded answer with citations..."):
                active_doc_name = st.session_state.doc_stats.get("filename", "")
                result = engine.generate_answer(
                    query=user_query,
                    retrieved_chunks=retrieved_chunks,
                    strict_grounding=strict_grounding,
                    filename=active_doc_name
                )
                answer_text = result["answer"]
                
                # Append latency note if chunks found
                if retrieved_chunks:
                    footer_note = f"\n\n*⚡ FAISS vector search completed in `{latency:.2f}ms` across `{len(engine.chunks)}` chunks.*"
                    full_display = answer_text + footer_note
                else:
                    full_display = answer_text
                    
                st.markdown(full_display)

                # Show expandable sources if available
                if retrieved_chunks and result.get("sources"):
                    with st.expander(f"🔍 View {len(retrieved_chunks)} Retrieved FAISS Chunks & Page Citations"):
                        for src in retrieved_chunks:
                            st.markdown(f"""
                            <div class="source-box">
                                <div class="source-header">
                                    <span class="badge-page">Page {src.get('page', '?')} • Chunk #{src.get('id', '?')}</span>
                                    <span class="badge-score">Similarity: {src.get('similarity_score', 0):.2%}</span>
                                </div>
                                <div>{src.get('text', '')}</div>
                            </div>
                            """, unsafe_allow_html=True)

            # Store in session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_display,
                "sources": result.get("sources", [])
            })

# Footer & Controls
if st.session_state.messages:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
