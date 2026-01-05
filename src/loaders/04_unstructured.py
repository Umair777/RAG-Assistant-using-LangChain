from langchain_community.document_loaders import UnstructuredMarkdownLoader
import nltk

def ensure_nltk():
    required = ["punkt", "punkt_tab"]
    for resource in required:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource)

def load_unstructured_md(path: str):
    ensure_nltk()
    loader = UnstructuredMarkdownLoader(path)
    docs = loader.load()
    return docs

if __name__ == "__main__":
    documents = load_unstructured_md("data/markdown-sample.md")
    print(f"Loaded {len(documents)} documents.")
    print(documents[0].page_content[:500])   