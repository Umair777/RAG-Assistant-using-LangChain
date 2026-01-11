from langchain_community.vectorstores import Chroma
import embeddings

def create_vector_store():
    docs, embedding_function, ids = embeddings.embed_and_store()

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding_function,
        ids=ids,
        collection_name="policies_collection",
        persist_directory="chroma_db"
    )

    vector_store.persist()
    print("Vector store created and persisted successfully.")
    for i in range(3):
        print(vector_store._collection.get(ids=str(i)))
        
    print(f"Total documents in collection: {vector_store._collection.count()}")
    return vector_store
    
    
    