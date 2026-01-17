from langchain_core.documents import Document

def add_metadata_to_document(text: str, metadata: dict):
    
    new_doc = Document(page_content=text,
                    metadata=metadata)

    print(f"Document content: {new_doc.page_content}")
    print(f"Document metadata: {new_doc.metadata}")
    new_chunk  = [new_doc]
    return new_chunk


    

