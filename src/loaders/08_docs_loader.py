from langchain_community.document_loaders import Docx2txtLoader

if __name__ == "__main__":
    loader = Docx2txtLoader("data/file_sample.docx")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    print("---- Document preview ----")
    print(documents[0].page_content[:500])