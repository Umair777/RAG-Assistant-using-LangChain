from langchain_core.documents import Document

def update_metadata_of_document(text, metadata: dict):
    update_chunk =  Document(
        page_content=text,
        metadata=metadata
        
    )
    return update_chunk