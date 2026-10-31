from rag_engine import RAGEngine

def test_pipeline():
    engine = RAGEngine(provider='local')
    chunks = engine.load_and_split_pdf('d:/ai/sample_document.pdf')
    print(f"Chunks loaded: {len(chunks)}")
    assert len(chunks) > 0, "No chunks created"

    ntotal = engine.build_vector_store()
    print(f"FAISS vectors stored: {ntotal}, dimension: {engine.embedding_dimension}")
    assert ntotal == len(chunks), "Vector count mismatch"

    query = "What is the optimal chunk size?"
    results = engine.retrieve(query, top_k=3)
    print(f"Retrieved {len(results)} chunks for query: '{query}'")
    for r in results:
        print(f"  - Rank #{r['rank']} | Page {r['page']} | Score: {r['similarity_score']:.3f} | Text: {r['text'][:70]}...")

    answer_data = engine.generate_answer(query, results)
    print("\nGenerated Output:")
    print(answer_data['answer'])
    print("\nSUCCESS! RAG pipeline completed end-to-end.")

if __name__ == "__main__":
    test_pipeline()
