from langchain_community.document_loaders import TextLoader

def LoadText(path:str):
    Loader = TextLoader(path)
    docs = Loader.load()
    return docs

if __name__ == "__main__":
    file_path = "new-Policies.txt"  # Replace with your text file path
    docs = LoadText(file_path)
    for i, doc in enumerate(docs):
        print(f"Document {i+1}: {doc.page_content[:100]}...")  # Print first 100 characters of each document

