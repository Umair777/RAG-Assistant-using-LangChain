from langchain_core.documents import Document

def add_metadata_to_document():
    text = "Instructlab is the best open source tool for fine-tuning a LLM."

    new_doc = Document(page_content=text,
                    metadata={
                            "source": "Instructlab blog",
                                "author": "Ashraf",
                                "document_created": "2024-06-01"
                    })


    print(f"Document content: {new_doc.page_content}")
    print(f"Document metadata: {new_doc.metadata}")
    new_chunk  = [new_doc]
    return new_chunk


    

