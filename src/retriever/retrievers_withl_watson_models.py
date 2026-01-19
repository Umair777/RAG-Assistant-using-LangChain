from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

from langchain_ibm import WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma


# -------- Text Splitter --------
def split_text(data, chunk_size=200, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(data)


# -------- Embeddings --------
def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }

    return WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr-v2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )


# -------- LLM --------
def llm():
    model_id = "mistralai/mistral-small-3-1-24b-instruct-2503"

    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }

    model = ModelInference(
        model_id=model_id,
        params=parameters,
        credentials={"url": "https://us-south.ml.cloud.ibm.com"},
        project_id="skills-network",
    )

    return WatsonxLLM(model=model)


# -------- Main --------
if __name__ == "__main__":
    loader = TextLoader("companypolicies.txt")
    txt_data = loader.load()
    print(f"Total documents loaded: {len(txt_data)}")

    chunks_txt = split_text(txt_data)

    vectordb = Chroma.from_documents(
        documents=chunks_txt,
        embedding=watsonx_embedding()
    )

    query = "email policy"
    retriever = vectordb.as_retriever()
    docs = retriever.invoke(query)

    print(f"\n========= Documents retrieved for query '{query}': {len(docs)}\n")
    for doc in docs:
        print(doc.page_content)
