from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter

if __name__ == "__main__":
    print("This module loads and prints the content of a text file.")
    with open("../../data/new-Policies.txt") as f:
        content = f.read()
        print(content[:500])
        
    text_splitter = CharacterTextSplitter(
        separator = "",
        chunk_size = 100,
        chunk_overlap = 20,
        length_function = len,
    )        
    new_doc = text_splitter.create_documents([content], metadatas=[{"source": "new-Policies.txt"}])
    print(f"Number of chunks created: {len(new_doc)}")
    print(f"first chunk content: {new_doc[0].page_content}")
    print(f"first chunk metadata: {new_doc[0].metadata}")
    print(f"first chunk source metadata: {new_doc[0].metadata['source']}")     