from langchain_community.document_loaders import TextLoader 

def load_text(pth: str):
    loader = TextLoader(pth, encoding='utf8')
    docs = loader.load()
    return docs

if __name__ == "__main__":
    documents = load_text("../../data/new-Policies.txt")
    print(f"Loaded {len(documents)} documents.")
    print(documents[0].page_content[:500])
    