"""
RAG Engine (Retrieval-Augmented Generation) Core Module
Implements:
1. Load & Split (PDF text extraction and chunking with page tracking)
2. Embed (Gemini, OpenAI, Groq, or Local Dense Vectorizer)
3. Store (FAISS IndexFlatIP cosine vector store)
4. Retrieve (Top-K similarity search with scores & latency tracking)
5. Answer (Strict grounded synthesis with citations & smart offline fallback)
6. Dynamic Document-Aware Question Suggestions
"""

import os
import re
import time
import warnings
from collections import Counter
import numpy as np
import faiss
from pypdf import PdfReader
from typing import List, Dict, Any, Tuple, Optional

# Suppress non-critical library deprecation notices for clean presentation
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Provider imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LocalDenseEmbedder:
    """
    High-speed, deterministic 256-dimensional semantic embedder.
    Enables instant offline demoing & FAISS vector search without requiring
    external API keys or network latency.
    """
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            words = re.findall(r'\b[a-zA-Z0-9_-]{2,}\b', text.lower())
            if not words:
                vec[0] = 1.0
            else:
                for idx, word in enumerate(words):
                    # Multi-hash projection into dense dimension
                    h1 = hash(word) % self.dim
                    h2 = hash(word + "_context") % self.dim
                    weight = 1.0 / (idx + 1)**0.12
                    vec[h1] += weight
                    vec[h2] += 0.6 * weight
                    # Character n-grams for morphological similarity
                    for i in range(len(word) - 2):
                        tri = word[i:i+3]
                        h3 = hash(tri) % self.dim
                        vec[h3] += 0.25
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])


class RAGEngine:
    def __init__(
        self,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.base_url = base_url
        
        self.chunks: List[Dict[str, Any]] = []
        self.raw_pages: List[Dict[str, Any]] = []
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.embedding_dimension: int = 0
        self.last_search_latency_ms: float = 0.0
        self.local_embedder = LocalDenseEmbedder(dim=256)
        self.document_title = ""
        self.key_topics: List[str] = []
        
        self._init_client()

    def _init_client(self):
        """Configure AI provider client."""
        if self.provider == "gemini" and self.api_key:
            if GEMINI_AVAILABLE:
                genai.configure(api_key=self.api_key)
        elif self.provider in ["openai", "custom"] and self.api_key:
            if OPENAI_AVAILABLE:
                self.openai_client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url if self.base_url else None
                )
        elif self.provider == "groq" and self.api_key:
            if OPENAI_AVAILABLE:
                self.openai_client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.groq.com/openai/v1"
                )

    # ---------------------------------------------------------
    # 1. LOAD & SPLIT
    # ---------------------------------------------------------
    def load_and_split_pdf(self, pdf_file) -> List[Dict[str, Any]]:
        """
        Extract text from uploaded PDF or file path, preserving page numbers,
        and segment into chunks of ~chunk_size characters with overlap.
        """
        reader = PdfReader(pdf_file)
        self.raw_pages = []
        
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            # Clean excessive spacing
            cleaned = re.sub(r'[ \t]+', ' ', text)
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
            if cleaned:
                self.raw_pages.append({
                    "page_num": page_idx + 1,
                    "text": cleaned,
                    "char_count": len(cleaned),
                    "word_count": len(cleaned.split())
                })

        # Analyze key topics from first few pages
        self._extract_document_profile()

        chunks: List[Dict[str, Any]] = []
        chunk_id = 0

        for page in self.raw_pages:
            page_num = page["page_num"]
            page_text = page["text"]
            start = 0
            text_len = len(page_text)
            
            while start < text_len:
                end = min(start + self.chunk_size, text_len)
                if end < text_len:
                    boundary = max(
                        page_text.rfind('. ', start, end),
                        page_text.rfind('\n', start, end),
                        page_text.rfind('; ', start, end),
                        page_text.rfind(' ', start, end)
                    )
                    if boundary != -1 and boundary > start + (self.chunk_size // 2):
                        end = boundary + 1

                chunk_str = page_text[start:end].strip()
                if chunk_str:
                    chunks.append({
                        "id": chunk_id,
                        "text": chunk_str,
                        "page": page_num,
                        "char_count": len(chunk_str),
                        "word_count": len(chunk_str.split()),
                        "char_range": (start, end)
                    })
                    chunk_id += 1

                start += max(1, self.chunk_size - self.chunk_overlap)
                if start >= text_len:
                    break

        self.chunks = chunks
        return chunks

    def _extract_document_profile(self):
        """Analyze prominent keywords and headings from the document."""
        if not self.raw_pages:
            return
        
        combined_header_text = " ".join([p["text"] for p in self.raw_pages[:4]])
        
        # Stopwords to filter out
        stopwords = {
            "this", "that", "with", "from", "have", "were", "which", "their", "there",
            "page", "chapter", "section", "about", "using", "these", "other", "into",
            "some", "such", "than", "then", "when", "more", "also", "each", "will"
        }
        
        # Extract meaningful nouns/keywords
        words = [
            w.capitalize() for w in re.findall(r'\b[A-Za-z0-9_-]{4,20}\b', combined_header_text)
            if w.lower() not in stopwords and not w.isdigit()
        ]
        
        top_counts = Counter(words).most_common(8)
        self.key_topics = [word for word, count in top_counts]
        
        # Determine likely document title
        first_lines = [l.strip() for l in self.raw_pages[0]["text"].split('\n') if len(l.strip()) > 8]
        if first_lines:
            self.document_title = first_lines[0][:60]
        else:
            self.document_title = "Uploaded Document"

    def generate_suggested_questions(self, filename: str = "") -> List[str]:
        """
        Dynamically generate 3 relevant, highly specific questions tailored to the uploaded document.
        """
        # If it's the demo RAG whitepaper
        if "sample" in filename.lower() or "rag" in filename.lower() or "whitepaper" in filename.lower():
            return [
                "What are the five pillars of RAG architecture?",
                "What is the optimal chunk size and overlap?",
                "What retrieval latency was achieved in benchmarks?"
            ]

        # Check for 8086 or Microprocessor documents
        full_text_sample = " ".join([p["text"] for p in self.raw_pages[:3]]).lower()
        if "8086" in full_text_sample or "microprocessor" in full_text_sample or "interfacing" in filename.lower():
            return [
                "What are the key memory interfacing techniques for the 8086 microprocessor?",
                "How does 8086 interface with RAM, ROM, and peripheral devices?",
                "What control signals and address registers are used during I/O operations?"
            ]

        # For generic documents, build dynamic questions from detected key topics
        if len(self.key_topics) >= 3:
            t1, t2, t3 = self.key_topics[0], self.key_topics[1], self.key_topics[2]
            return [
                f"What are the primary functions and characteristics of {t1}?",
                f"How does the document describe {t2} in relation to {t3}?",
                f"What are the key technical specifications or steps explained in the text?"
            ]
        elif len(self.key_topics) >= 1:
            t1 = self.key_topics[0]
            return [
                f"What is the main purpose and function of {t1}?",
                "What key findings or procedures are detailed in this document?",
                "Summarize the main conclusions and guidelines presented."
            ]

        return [
            "What is the executive summary or purpose of this document?",
            "What are the main topics and sections covered?",
            "What are the key conclusions or recommendations provided?"
        ]

    # ---------------------------------------------------------
    # 2. EMBED
    # ---------------------------------------------------------
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate dense embeddings for a list of text strings.
        Supports Gemini, OpenAI, or falls back to Local Embedder.
        """
        # Google Gemini
        if self.provider == "gemini" and self.api_key and GEMINI_AVAILABLE:
            try:
                emb_model = "models/text-embedding-004"
                res = genai.embed_content(
                    model=emb_model,
                    content=texts,
                    task_type="retrieval_document"
                )
                embeddings = np.array(res['embedding'], dtype=np.float32)
                faiss.normalize_L2(embeddings)
                return embeddings
            except Exception as e:
                print(f"[Gemini Embed Note]: {e}. Using fast local embedder.")

        # OpenAI
        elif self.provider in ["openai", "custom"] and self.api_key and OPENAI_AVAILABLE:
            try:
                emb_model = "text-embedding-3-small"
                response = self.openai_client.embeddings.create(
                    input=texts,
                    model=emb_model
                )
                embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)
                faiss.normalize_L2(embeddings)
                return embeddings
            except Exception as e:
                print(f"[OpenAI Embed Note]: {e}. Using fast local embedder.")

        # Local Deterministic Embedder
        embeddings = self.local_embedder.embed_texts(texts)
        faiss.normalize_L2(embeddings)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single search query."""
        if self.provider == "gemini" and self.api_key and GEMINI_AVAILABLE:
            try:
                emb_model = "models/text-embedding-004"
                res = genai.embed_content(
                    model=emb_model,
                    content=query,
                    task_type="retrieval_query"
                )
                vec = np.array([res['embedding']], dtype=np.float32)
                faiss.normalize_L2(vec)
                return vec
            except Exception as e:
                pass

        elif self.provider in ["openai", "custom"] and self.api_key and OPENAI_AVAILABLE:
            try:
                emb_model = "text-embedding-3-small"
                response = self.openai_client.embeddings.create(
                    input=[query],
                    model=emb_model
                )
                vec = np.array([response.data[0].embedding], dtype=np.float32)
                faiss.normalize_L2(vec)
                return vec
            except Exception as e:
                pass

        vec = self.local_embedder.embed_query(query)
        faiss.normalize_L2(vec)
        return vec

    # ---------------------------------------------------------
    # 3. STORE (FAISS Vector Database)
    # ---------------------------------------------------------
    def build_vector_store(self, chunks: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        Embed all chunks and store them in a FAISS IndexFlatIP vector database.
        Returns the number of indexed vectors.
        """
        if chunks:
            self.chunks = chunks

        if not self.chunks:
            raise ValueError("No chunks to index. Please load and split a document first.")

        texts = [c["text"] for c in self.chunks]
        embeddings = self.generate_embeddings(texts)

        self.embedding_dimension = embeddings.shape[1]
        
        # FAISS IndexFlatIP (Inner Product = Cosine Similarity for normalized vectors)
        index = faiss.IndexFlatIP(self.embedding_dimension)
        index.add(embeddings)
        
        self.faiss_index = index
        return index.ntotal

    # ---------------------------------------------------------
    # 4. RETRIEVE
    # ---------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Embed query and retrieve top_k most similar chunks with scores and latency tracking.
        """
        if self.faiss_index is None or not self.chunks:
            return []

        start_time = time.perf_counter()
        query_vec = self.embed_query(query)
        top_k = min(top_k, len(self.chunks))
        
        distances, indices = self.faiss_index.search(query_vec, top_k)
        self.last_search_latency_ms = (time.perf_counter() - start_time) * 1000
        
        retrieved_chunks = []
        for rank, (chunk_idx, score) in enumerate(zip(indices[0], distances[0])):
            if chunk_idx != -1 and chunk_idx < len(self.chunks):
                chunk_data = self.chunks[chunk_idx].copy()
                chunk_data["similarity_score"] = float(score)
                chunk_data["rank"] = rank + 1
                retrieved_chunks.append(chunk_data)

        return retrieved_chunks

    # ---------------------------------------------------------
    # 5. ANSWER (LLM Grounded Synthesis)
    # ---------------------------------------------------------
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        strict_grounding: bool = True,
        filename: str = ""
    ) -> Dict[str, Any]:
        """
        Synthesize an answer using LLM strictly from retrieved context chunks.
        Includes topic mismatch detection and structured offline answers.
        """
        if not retrieved_chunks:
            return {
                "answer": "I could not find any relevant information in the uploaded document to answer your question.",
                "sources": [],
                "confidence": 0.0,
                "provider": "None"
            }

        # Check for obvious topic mismatch (e.g. asking about "RAG" on an 8086 PDF)
        query_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
        combined_retrieved = " ".join([c["text"].lower() for c in retrieved_chunks])
        overlap_count = sum(1 for term in query_terms if term in combined_retrieved)
        
        # If user asked something with 0 keyword match and low similarity
        if overlap_count == 0 and retrieved_chunks[0]["similarity_score"] < 0.28:
            doc_context = f"**{filename}**" if filename else "the uploaded document"
            topics_hint = ", ".join(self.key_topics[:4]) if self.key_topics else "the topics discussed in the document"
            suggested = self.generate_suggested_questions(filename)
            
            mismatch_msg = (
                f"⚠️ **Topic Not Found in Current Document**\n\n"
                f"You asked: *\"{query}\"*\n\n"
                f"However, {doc_context} appears to be about **{topics_hint}** and does not mention this topic.\n\n"
                f"💡 **Try asking questions related to {doc_context}:**\n"
                f"1. 👉 *{suggested[0]}*\n"
                f"2. 👉 *{suggested[1]}*\n"
                f"3. 👉 *{suggested[2]}*"
            )
            return {
                "answer": mismatch_msg,
                "sources": [],
                "model_used": "Topic Guard",
                "provider": "Document Validation"
            }

        # Format context with citations
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            block = f"[Chunk #{chunk['id']} | Page {chunk['page']} | Relevance: {chunk['similarity_score']:.2f}]\n{chunk['text']}"
            context_blocks.append(block)
        
        context_str = "\n\n---\n\n".join(context_blocks)

        system_instruction = (
            "You are an expert, highly precise Document Q&A Assistant. "
            "Your task is to answer the user's question using ONLY the provided document context below.\n"
            "STRICT RULES:\n"
            "1. Answer ONLY from the provided context. If the answer cannot be found in the context, explicitly say: "
            "'I could not find the answer in the uploaded document.'\n"
            "2. Never make up facts or speculate outside the context.\n"
            "3. Mention source page numbers (e.g., [Page X]) when referencing specific facts.\n"
            "4. Provide a thorough, well-structured, clear explanation with bullet points when appropriate."
        )

        user_prompt = f"DOCUMENT CONTEXT:\n{context_str}\n\nUSER QUESTION:\n{query}\n\nANSWER:"

        # 1. Google Gemini Synthesis
        if self.provider == "gemini" and self.api_key and GEMINI_AVAILABLE:
            try:
                target_model = self.model_name or "gemini-1.5-flash"
                model = genai.GenerativeModel(
                    model_name=target_model,
                    system_instruction=system_instruction
                )
                response = model.generate_content(user_prompt)
                return {
                    "answer": response.text,
                    "sources": retrieved_chunks,
                    "model_used": target_model,
                    "provider": "Google Gemini"
                }
            except Exception as e:
                print(f"[Gemini Answer Error]: {e}")

        # 2. OpenAI or Groq Synthesis
        elif self.provider in ["openai", "groq", "custom"] and self.api_key and OPENAI_AVAILABLE:
            try:
                if self.provider == "groq":
                    target_model = self.model_name or "llama-3.3-70b-versatile"
                else:
                    target_model = self.model_name or "gpt-4o-mini"
                    
                completion = self.openai_client.chat.completions.create(
                    model=target_model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2
                )
                return {
                    "answer": completion.choices[0].message.content,
                    "sources": retrieved_chunks,
                    "model_used": target_model,
                    "provider": "Groq (Llama 3)" if self.provider == "groq" else "OpenAI"
                }
            except Exception as e:
                print(f"[LLM Answer Error]: {e}")

        # 3. Rich Local Extractive Grounded Synthesis (When running in Free Zero-Key Mode)
        # Pull best sentences across top chunks and organize cleanly
        extracted_sections = []
        for idx, chunk in enumerate(retrieved_chunks[:3]):
            sentences = re.split(r'(?<=[.!?\n])\s+', chunk["text"])
            matched_sentences = []
            for s in sentences:
                s_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', s.lower()))
                if query_terms.intersection(s_words) and len(s.strip()) > 20:
                    matched_sentences.append(s.strip())
            
            if matched_sentences:
                passage = " ".join(matched_sentences[:2])
            else:
                passage = chunk["text"][:220].strip() + "..."
                
            extracted_sections.append(
                f"**From [Page {chunk['page']}] (Chunk #{chunk['id']} — Similarity `{chunk['similarity_score']:.1%}`):**\n"
                f"> \"{passage}\""
            )

        passages_text = "\n\n".join(extracted_sections)

        structured_offline_answer = (
            f"### 📑 Retrieved Document Evidence for: *\"{query}\"*\n\n"
            f"{passages_text}\n\n"
            f"---\n"
            f"ℹ️ **Mode: Local FAISS Vector Search** *(Zero API Key Required)*\n"
            f"The above passages were retrieved directly from your document based on semantic vector similarity. "
            f"To get a **full AI conversational synthesis** (where the model writes a complete essay/explanation), "
            f"simply paste your free Google Gemini or Groq API key in the top banner or sidebar."
        )
        
        return {
            "answer": structured_offline_answer,
            "sources": retrieved_chunks,
            "model_used": "FAISS Semantic Retrieval",
            "provider": "Local Vector Database"
        }
