from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import load_data
import split_data

def embed_and_store():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)

    documents = load_data.load_text_file("new-Policies.txt")
    text = documents[0].page_content

    split_texts = split_data.split_data(text)

    docs = [
        Document(page_content=chunk, metadata={"chunk_id": i})
        for i, chunk in enumerate(split_texts)
    ]

    ids = [str(i) for i in range(len(docs))]

    return docs, embedding_function, ids
