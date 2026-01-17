from langchain.vectorstores import FAISS
import embeddings

def faiss_vector_store():
    docs, embedding_function, ids = embeddings.embed_and_store()
    vector_store = FAISS.from_documents(
        documents=docs,
        embedding=embedding_function,
        ids=ids,
       
        )
    # vector_store.add
    print("FAISS Vector store created successfully.")
    print(f"Total documents in FAISS store: {vector_store.index.ntotal}")
    print("Total document IDs in FAISS store:", len(list(vector_store.docstore._dict.keys())))
    return vector_store