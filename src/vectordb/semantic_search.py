import vector_store
if __name__ == "__main__":
    query = "Email policy"
    vector_store_instance = vector_store.create_vector_store()
    docs = vector_store_instance.similarity_search(query)
    print(f"Documents retrieved for query '{query}': {len(docs)}")
    # print(docs[0].page_content[:500])
    print(docs)