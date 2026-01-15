import chroma_vector_store

if __name__ == "__main__":
    query = "Email policy"
    chroma_store_instance = chroma_vector_store.chromadb_vector_store()
    docs = chroma_store_instance.similarity_search(query)
    print(f"Documents retrieved for query '{query}': {len(docs)}")
    # print(docs[0].page_content[:500])
    print(docs)