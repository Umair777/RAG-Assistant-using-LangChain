from langchain_community.document_loaders import TextLoader

def LoadText(path:str):
    Loader = TextLoader(path)
    docs = Loader.load()
    return docs

