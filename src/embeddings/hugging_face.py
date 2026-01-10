from langchain_community.embeddings import HuggingFaceEmbeddings

if __name__ == "__main__":
    model_name = "sentence-transformers/all-mpnet-base-v2"
    huggingface_embedding = HuggingFaceEmbeddings(model_name=model_name)
    query = "How are you?"
    query_result = huggingface_embedding.embed_query(query)
    print(f"Embedding result: {query_result}")