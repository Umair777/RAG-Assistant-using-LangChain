
import faiss_db
if __name__ == "__main__":
    query = "Email policy"
    faiss_store_instance = faiss_db.faiss_vector_store()
    docs = faiss_store_instance.similarity_search(query)
    print(f"=========Documents retrieved for query '{query}': {len(docs)}")
    print(list(faiss_store_instance.docstore._dict.keys())[:5])  # Print first 5 document IDs in the docstore
    doc = faiss_store_instance.docstore.search("215")
    print(doc)
    print("Total documents in FAISS store:", faiss_store_instance.index.ntotal)
    
    # print(docs[0].page_content[:500]) 
    # print(faiss_store_instance._collection.get(ids=['215']))
    # print(docs) 
    
    