from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

from langchain_ibm import WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline


# -------- Text Splitter --------
def split_text(data, chunk_size=200, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(data)


# -------- Embeddings --------
def my_embedding_model():
    return SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# -------- LLM --------
# def llm():
#     model_id = "mistralai/Mistral-7B-Instruct-v0.2"

#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16,
#         device_map="auto"
#     )

#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=256,
#         temperature=0.3,
#         do_sample=True
#     )

#     return HuggingFacePipeline(pipeline=pipe)


# -------- Main --------
if __name__ == "__main__":
    loader = TextLoader("companypolicies.txt")
    txt_data = loader.load()
    print(f"Total documents loaded: {len(txt_data)}")

    chunks_txt = split_text(txt_data)

    vectordb = Chroma.from_documents(
        documents=chunks_txt,
        embedding=my_embedding_model()
    )

    query = "email policy"
    retriever = vectordb.as_retriever()
    docs = retriever.invoke(query)

    print(f"\n========= Documents retrieved for query '{query}': {len(docs)}\n")
    for doc in docs:
        print(doc.page_content)
        
    
