from langchain_community.document_loaders import PyPDFLoader

def load_pdf(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs

if __name__ == "__main__":
    documents = load_pdf("data/instructlab.pdf")
    print(f"Loaded {len(documents)} documents.")
    print("---- Page 1 preview ----")
    print(documents[0].page_content[:500])
    