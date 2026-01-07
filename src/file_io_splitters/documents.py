from langchain_core.documents import Document

if __name__ == "__main__":
    doc = Document(page_content = """This is a sample document and I have working very hard on this
                   It has taken me a lot of time to write this.
                   I hope you like it.""",
                   metadata = {
                       "my_document_id": "001",
                        "author": "Ashraf",
                        "document_created": "2026-01-01"
                        }
                   )
    print(f"Document content: {doc.page_content}")
    print(f"Document metadata: {doc.metadata}")
    print(f"Document ID: {doc.metadata['my_document_id']}")
    print(f"Document Authon: {doc.metadata['author']}")
    