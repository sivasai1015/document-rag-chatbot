"""
Script to generate a sample high-quality PDF document for Document Q&A RAG testing.
Topic: "Next-Generation Autonomous AI Agents & Vector Architectures"
"""
from fpdf import FPDF
import os

class SampleReportPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 116, 139)
        self.cell(0, 8, 'RESEARCH WHITE PAPER  |  DEEP DIVE INTO MODERN RAG & VECTOR SEARCH', 0, 1, 'R')
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_sample_pdf(output_path="sample_document.pdf"):
    pdf = SampleReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Page 1: Title and Executive Summary
    pdf.add_page()
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    
    # Document Title
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(15, 23, 42)
    pdf.multi_cell(0, 10, 'Autonomous AI Systems & Retrieval-Augmented Generation (RAG)')
    pdf.ln(3)
    
    pdf.set_font('Helvetica', 'I', 11)
    pdf.set_text_color(79, 70, 229)
    pdf.cell(0, 8, 'Technical Architecture, Vector Embeddings, and Production Scaling', 0, 1, 'L')
    pdf.ln(5)
    
    # Section 1
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, '1. Executive Summary', 0, 1, 'L')
    pdf.ln(2)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(51, 65, 85)
    summary_text = (
        "Retrieval-Augmented Generation (RAG) is an architectural pattern that enhances the output of Large "
        "Language Models (LLMs) by referencing an authoritative knowledge base outside of its training data "
        "before generating a response. Traditional LLMs are prone to hallucinations and lack awareness of private, "
        "enterprise, or recent documentation. By introducing a real-time retrieval pipeline consisting of "
        "document chunking, dense vector representations, and high-performance vector indexes like FAISS (Facebook AI "
        "Similarity Search), applications achieve verified, grounded answers with zero retraining costs."
    )
    pdf.multi_cell(0, 6, summary_text)
    pdf.ln(4)
    
    # Section 2: Core Components of Modern RAG
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, '2. The Five Pillars of RAG Architecture', 0, 1, 'L')
    pdf.ln(2)
    
    pillars = [
        ("Pillar 1: Ingestion & Text Splitting", 
         "Documents such as PDFs contain unstructured layout blocks. Text is parsed page-by-page and segmented into semantic chunks (typically 500 characters with 10-20% overlap). Overlap preserves contextual continuity across chunk boundaries."),
        ("Pillar 2: Dense Semantic Embedding", 
         "Each text chunk is converted into high-dimensional dense vectors (e.g., 768 or 1536 floating-point values) via transformer embedding models. Semantic proximity is preserved mathematically in vector space."),
        ("Pillar 3: High-Performance Vector Storage (FAISS)", 
         "FAISS indexes vectors using Euclidean (L2) or Inner Product (Cosine) metrics. By normalizing embeddings, IndexFlatIP delivers millisecond similarity searches even across millions of vectors."),
        ("Pillar 4: Contextual Retrieval", 
         "User questions are projected into the same embedding space. The top 3 to 4 nearest neighbors are retrieved, filtering out irrelevant noise while maximizing relevance."),
        ("Pillar 5: Constrained LLM Synthesis", 
         "The retrieved passages are injected into a strict system prompt instructing the model to answer solely using the provided context, preventing hallucinations.")
    ]
    
    for title, desc in pillars:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(67, 56, 202)
        pdf.cell(0, 6, title, 0, 1, 'L')
        pdf.set_font('Helvetica', '', 9.5)
        pdf.set_text_color(51, 65, 85)
        pdf.multi_cell(0, 5, desc)
        pdf.ln(2)
        
    # Page 2: Advanced Mechanics & Benchmark Metrics
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, '3. Empirical Benchmarks & Performance Metrics', 0, 1, 'L')
    pdf.ln(3)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(51, 65, 85)
    benchmarks_desc = (
        "In production evaluations, the combination of FAISS IndexFlatIP with normalized 768-dimensional "
        "embeddings yielded an average retrieval latency of 4.2 milliseconds across 50,000 document chunks. "
        "The hallucination rate was reduced from 28.4% in naive zero-shot generation to under 1.2% when strict "
        "contextual constraints and source citation requirements were enforced."
    )
    pdf.multi_cell(0, 6, benchmarks_desc)
    pdf.ln(5)
    
    # Key stats bullet points
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, 'Key Findings & Best Practices:', 0, 1, 'L')
    pdf.ln(1)
    
    bullet_points = [
        "- Optimal Chunk Size: 500 characters with 75 character overlap prevents context severance.",
        "- Top-K Selection: 3 to 4 retrieved chunks provide sufficient coverage without overloading prompt context windows.",
        "- Cosine Similarity vs L2: Normalized vectors using Inner Product (IP) achieve faster calculation and bounded [-1, 1] similarity scoring.",
        "- Grounding Directive: Explicit instructions ('Answer only using the provided text; if absent, reply unable to find') eliminate fabrication.",
        "- Page Tracking: Preserving metadata (page number, chunk ID) empowers users to verify claims against original source pages."
    ]
    
    for bp in bullet_points:
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(51, 65, 85)
        pdf.multi_cell(0, 6, bp)
        pdf.ln(1)
        
    pdf.ln(4)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, '4. Security, Privacy & Future Outlook', 0, 1, 'L')
    pdf.ln(2)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(51, 65, 85)
    outlook = (
        "As organizations handle sensitive intellectual property, on-premise and client-side vector search "
        "ensures that raw document data never leaks into general training corpuses. Looking forward into 2026, "
        "hybrid retrieval (combining dense vector search with sparse BM25 lexical search) and recursive graph RAG "
        "represent the next frontier in autonomous reasoning systems."
    )
    pdf.multi_cell(0, 6, outlook)
    
    pdf.output(output_path)
    print(f"Sample PDF created successfully at: {output_path}")

if __name__ == "__main__":
    generate_sample_pdf("d:/ai/sample_document.pdf")
