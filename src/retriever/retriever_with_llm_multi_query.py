



from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
import torch

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
def llm():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.3,
        do_sample=True
    )

    return HuggingFacePipeline(pipeline=pipe)


# -------- Main --------
if __name__ == "__main__":
    loader = PyPDFLoader("instructlab.pdf")
    PyPDFLoader_data = loader.load()
    print(f"Total documents loaded: {len(PyPDFLoader_data)}")

    chunks_txt = split_text(PyPDFLoader_data)
    print(f"Total chunks created: {len(chunks_txt)}")

    vectordb = Chroma.from_documents(
        documents=chunks_txt,
        embedding=my_embedding_model()
    )
    query = "What does the paper say about langchain?"

    retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm()
    )
    results = retriever.get_relevant_documents(query)

    print(f"Total relevant documents retrieved: {len(results)}")
    for i, doc in enumerate(results):
        print(f"\n--- Document {i+1} ---")
        print(doc.page_content)        
    
''''
OUTPUT:

ut 3 were given
Failed to send telemetry event ClientCreateCollectionEvent: capture() takes 1 positional argument but 3 were given
config.json: 100%|█████████████████████████████████████| 596/596 [00:00<00:00, 2.29MB/s]
`torch_dtype` is deprecated! Use `dtype` instead!
model.safetensors.index.json: 25.1kB [00:00, 99.7MB/s]
model-00001-of-00003.safetensors: 100%|████████████| 4.94G/4.94G [44:47<00:00, 1.84MB/s]
model-00003-of-00003.safetensors: 100%|████████████| 4.54G/4.54G [44:59<00:00, 1.68MB/s]
model-00002-of-00003.safetensors: 100%|████████████| 5.00G/5.00G [45:40<00:00, 1.82MB/s]
Fetching 3 files: 100%|██████████████████████████████████| 3/3 [45:40<00:00, 913.66s/it]
Loading checkpoint shards: 100%|██████████████████████████| 3/3 [00:34<00:00, 11.64s/it]
generation_config.json: 100%|██████████████████████████| 111/111 [00:00<00:00, 91.5kB/s]
Some parameters are on the meta device because they were offloaded to the disk and cpu.
Device set to use cuda:0
/home/ashraf/llm/RAG-Assistant-using-LangChain/src/retriever/retriever_with_llm_multi_query.py:51: LangChainDeprecationWarning: The class `HuggingFacePipeline` was deprecated in LangChain 0.0.37 and will be removed in 0.3. An updated version of the class exists in the from rom langchain-huggingface package and should be used instead. To use it run `pip install -U from rom langchain-huggingface` and import as `from from rom langchain_huggingface import llms import HuggingFacePipeline`.
  return HuggingFacePipeline(pipeline=pipe)
/home/ashraf/llm/RAG-Assistant-using-LangChain/src/retriever/retriever_with_llm_multi_query.py:72: LangChainDeprecationWarning: The method `BaseRetriever.get_relevant_documents` was deprecated in langchain-core 0.1.46 and will be removed in 1.0. Use invoke instead.
  results = retriever.get_relevant_documents(query)
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
Failed to send telemetry event CollectionQueryEvent: capture() takes 1 positional argument but 3 were given
Total relevant documents retrieved: 31

--- Document 1 ---
task description in the form of an natural language instuction (e.g. Summarize the following news
article in 2 lines: {News article }) and the model is trained to maximize the likelihood of the pro-

--- Document 2 ---
and Ryan Lowe. Training language models to follow instructions with human feedback, 2022.
9

--- Document 3 ---
Hannaneh Hajishirzi. Self-Instruct: Aligning Language Models with Self-Generated Instructions,
May 2023.
Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, An-

--- Document 4 ---
Steinhardt. Measuring massive multitask language understanding. In International Conference
on Learning Representations , 2020.

--- Document 5 ---
a small number of handwritten human seed instructions as input to bootstrapping process to generate
a large number of samples using an LLM’s own generation abilities. Taori et al. (2023) built upon

--- Document 6 ---
modes), the teacher model is guaranteed to generate synthetic data for each of the tasks (purple).
With the above insight, we now introduce two new synthetic data generation (SDG) methods in LAB

--- Document 7 ---
and incorporating principles in the generation prompt to promote diversity in the generated instruc-

--- Document 8 ---
Similar to LAB, concurrent work, GLAN (Li et al., 2024), employs a semi-automatic approach to
synthetic data generation that uses a human-curated taxonomy to generate instruction tuning data

--- Document 9 ---
tics, etc.; see the sub-tree for knowledge in Figure 1 as an example. Each domain has a collection of
documents and a sample set of domain-specific questions and answers. This organization allows for

--- Document 10 ---
better control over the licensing of text documents. As described in the next section, only the doc-
uments with permissible licenses are selected for synthetic data generation, excluding knowledge

--- Document 11 ---
generates queries that adhere to specific principles and thoroughly explore the targeted
domain, enhancing the comprehensiveness of the generated content.

--- Document 12 ---
model is provided a knowledge source in the form of documents, manuals, and books on the target
subject to ground the generated instruction data into a reliable source thus avoiding dependence on

--- Document 13 ---
high-quality, contextually appropriate questions move forward in the process.
3.Generating responses: The teacher model, functioning as a response generator in this

--- Document 14 ---
* The questions should be clear and human-like.
* The questions should be diverse and cover a wide range of topics.
* The questions should not be template-based or generic, it should be very diverse.

--- Document 15 ---
using a specialized prompt (see Figure 3 for an example) to leverage its knowledge and
create diverse questions. By iterating through each leaf node of a taxonomy, the teacher

--- Document 16 ---
* Always generate safe and respectful content. Do not generate content that is harmful, abusive, or
offensive.
* Always generate content that is factually accurate and relevant to the prompt.

--- Document 17 ---
edge and foundational skills, synergistically, to answer complex queries from users. For instance, the
model’s ability to write a company-wide email sharing insights about the company’s performance

--- Document 18 ---
LAB: L ARGE -SCALE ALIGNMENT FOR CHATBOTS
MIT-IBM Watson AI Lab and IBM Research
Shivchander Sudalairaj∗
Abhishek Bhandwaldar∗
Aldo Pareja∗
Kai Xu
David D. Cox
Akash Srivastava∗,†

--- Document 19 ---
Akash Srivastava∗,†
*Equal Contribution, †Corresponding Author
ABSTRACT
This work introduces LAB (Large-scale Alignment for chatBots), a novel method-

--- Document 20 ---
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop
questions via single-hop question composition, 2022.

--- Document 21 ---
3: It means the answer is a perfect answer from an AI Assistant. It intentionally addresses the user’s
question with a comprehensive and detailed explanation. It demonstrates expert knowledge in the

--- Document 22 ---
models.
2

--- Document 23 ---
tasks, each of the tasks are guaranteed to be well represented in the prompts (purple). Second, when
4

--- Document 24 ---
8

--- Document 25 ---
arXiv:2305.15717 , 2023.
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob

--- Document 26 ---
Sanseviero, Alexander M. Rush, and Thomas Wolf. Zephyr: Direct Distillation of LM Alignment,
October 2023.
Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and

--- Document 27 ---
1 I NTRODUCTION
Large language models (LLMs) have achieved remarkable levels of success in various natural lan-

--- Document 28 ---
Ahmed Awadallah. Orca: Progressive learning from complex explanation traces of gpt-4, 2023.
Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong

--- Document 29 ---
that leverage the taxonomy to guide the data generation process. The first one is targeted for skills

--- Document 30 ---
the appropriate branch and attaching 1–3 examples.
Knowledge The knowledge branch in the taxonomy is first divided based on document types like

--- Document 31 ---
that don’t meet predefined principles, including relevance to the domain, potential harm,
or questions beyond a language model’s answering capabilities. This ensures that only
'''