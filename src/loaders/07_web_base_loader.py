from langchain_community.document_loaders import WebBaseLoader

def load_web_page(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

if __name__ == "__main__":
    url = "https://www.ibm.com/topics/langchain"
    documents = load_web_page(url)
    print(f"Loaded {len(documents)} documents from {url}.")
    print("---- Document preview ----")
    print(documents)
    
    
    # print(documents[0].page_content[:500])
