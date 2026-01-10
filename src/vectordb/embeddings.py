from langchain_community.embeddings import HuggingFaceEmbeddings
import load_data
import split_data


if __name__ == "__main__":
    model_name = "sentence-transformers/all-mpnet-base-v2"
    huggingface_embedding = HuggingFaceEmbeddings(model_name=model_name)
    # query = "How are you?"
    data = load_data.load_text_file("new-Policies.txt")
    print(f"Loaded {len(data)} documents.")
    print(data[0].page_content[:500])
    split_texts = split_data.split_data(data[0].page_content)
    print(f"Split into {len(split_texts)} chunks.")
    print(split_texts[0])
    query_result = huggingface_embedding.embed_query(split_texts[0])
    print(f"Embedding result: {query_result}")