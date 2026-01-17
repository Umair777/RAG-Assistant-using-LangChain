
# from xml.dom.minidom import Document
import faiss_db
import add_data_vector
from langchain_core.documents import Document
if __name__ == "__main__":
    query = "Email policy"
    faiss_store_instance = faiss_db.faiss_vector_store()
    docs = faiss_store_instance.similarity_search(query)
    print(f"=========Documents retrieved for query '{query}': {len(docs)}")
    print(list(faiss_store_instance.docstore._dict.keys())[:5])  # Print first 5 document IDs in the docstore
    doc = faiss_store_instance.docstore.search("215")
    print(doc)
    print("Total documents in FAISS store:", faiss_store_instance.index.ntotal)
    doc_id = "93"
    new_doc = Document(
    page_content="Instructlab is the best open source tool for fine-tuning a LLM.",
    metadata={
        "source": "Instructlab blog",
        "author": "Ashraf",
        "document_created": "2024-06-01",
        "id": doc_id
    })
    faiss_store_instance.add_documents([new_doc], ids=[doc_id])
    print("Document added successfully.") 
    print("Updated total documents in FAISS store:", faiss_store_instance.index.ntotal)
    new_doc = faiss_store_instance.docstore.search("93")
    print(new_doc)
    update_chunk = add_data_vector.add_metadata_to_document(
        "Instructlab is a perfect open source tool for fine-tuning a LLM.",
        {
            "source": "ibm.com",
            "page": 1
        }
    )
    print('This is the old document data before updating:')
    faiss_store_instance.delete(ids=["93"])
    print("Document with ID '93' deleted successfully.")
    print("Total documents in FAISS store after deletion:", faiss_store_instance.index.ntotal)
    updated_doc = Document(
    page_content="Instructlab is a perfect open source tool for fine-tuning a LLM.",
    metadata={
        "source": "ibm.com",
        "page": 1,
        "id": "93"
    }
)
faiss_store_instance.add_documents([updated_doc], ids=["93"])
print("Document with ID '93' updated successfully.")
print("Total documents in FAISS store after updating:", faiss_store_instance.index.ntotal)
   
    
    
    