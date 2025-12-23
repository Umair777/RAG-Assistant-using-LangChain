"""
RAG Inference Module
Implements the RAG pipeline for question answering
"""

from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from .vector_store import VectorStoreManager


class RAGInference:
    """RAG inference engine for question answering"""
    
    DEFAULT_PROMPT_TEMPLATE = """You are an AI assistant helping users find information from documents.
Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Provide detailed and accurate answers based on the retrieved information.

Context:
{context}

Question: {question}

Answer:"""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize RAG inference engine
        
        Args:
            vector_store_manager: Vector store manager instance
            llm_model: LLM model name
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            openai_api_key: OpenAI API key
        """
        self.vector_store_manager = vector_store_manager
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=openai_api_key
        )
        
        # Create prompt template
        self.prompt = PromptTemplate(
            template=self.DEFAULT_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Initialize QA chain
        self.qa_chain = None
    
    def create_qa_chain(
        self,
        chain_type: str = "stuff",
        k: int = 4,
        return_source_documents: bool = True
    ):
        """
        Create a QA chain with the vector store
        
        Args:
            chain_type: Type of chain ('stuff', 'map_reduce', 'refine', 'map_rerank')
            k: Number of documents to retrieve
            return_source_documents: Whether to return source documents
        """
        retriever = self.vector_store_manager.as_retriever(
            search_kwargs={"k": k}
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=return_source_documents,
            chain_type_kwargs={"prompt": self.prompt}
        )
    
    def query(
        self,
        question: str,
        k: int = 4
    ) -> Dict[str, Any]:
        """
        Query the RAG system with a question
        
        Args:
            question: Question to ask
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer and source documents
        """
        if self.qa_chain is None:
            self.create_qa_chain(k=k)
        
        result = self.qa_chain({"query": question})
        
        return {
            'question': question,
            'answer': result['result'],
            'source_documents': result.get('source_documents', [])
        }
    
    def query_with_retrieval_details(
        self,
        question: str,
        k: int = 4
    ) -> Dict[str, Any]:
        """
        Query with detailed retrieval information
        
        Args:
            question: Question to ask
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer, sources, and retrieval scores
        """
        # Get relevant documents with scores
        docs_with_scores = self.vector_store_manager.similarity_search_with_score(
            query=question,
            k=k
        )
        
        # Query the QA chain
        result = self.query(question, k=k)
        
        # Add score information
        result['retrieval_details'] = [
            {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'relevance_score': float(score)
            }
            for doc, score in docs_with_scores
        ]
        
        return result
    
    def batch_query(
        self,
        questions: List[str],
        k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch
        
        Args:
            questions: List of questions
            k: Number of documents to retrieve per question
            
        Returns:
            List of result dictionaries
        """
        results = []
        for question in questions:
            try:
                result = self.query(question, k=k)
                results.append(result)
            except Exception as e:
                results.append({
                    'question': question,
                    'answer': f"Error: {str(e)}",
                    'source_documents': []
                })
        
        return results
    
    def set_custom_prompt(self, template: str, input_variables: List[str]):
        """
        Set a custom prompt template
        
        Args:
            template: Prompt template string
            input_variables: List of input variable names
        """
        self.prompt = PromptTemplate(
            template=template,
            input_variables=input_variables
        )
        # Recreate QA chain with new prompt
        self.qa_chain = None
    
    def update_inference_params(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Update inference parameters
        
        Args:
            temperature: New temperature value
            max_tokens: New max tokens value
        """
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        
        # Recreate LLM with new parameters
        self.llm = ChatOpenAI(
            model_name=self.llm_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Reset QA chain to use new LLM
        self.qa_chain = None
    
    def get_inference_info(self) -> dict:
        """
        Get information about the inference configuration
        
        Returns:
            Dictionary with configuration details
        """
        return {
            'llm_model': self.llm_model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'chain_initialized': self.qa_chain is not None
        }
