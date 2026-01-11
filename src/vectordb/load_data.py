from langchain_community.document_loaders import TextLoader

def load_text_file(file_path: str):
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents

# if __name__ == "__main__":
#     file_path = "new-Policies.txt"  # Replace with your text file path
#     docs = load_text_file(file_path)
#     for i, doc in enumerate(docs):
#         print(f"Document {i+1}: {doc.page_content[:100]}...")  # Print first 100 characters of each document