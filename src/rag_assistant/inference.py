"""
RAG Inference Module
Implements the RAG pipeline for question answering
"""

from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=openai_api_key
        )
        
        # Create prompt template
        self.prompt = PromptTemplate(
            template=self.DEFAULT_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Initialize chain
        self.chain = None
        self._retriever = None
    
    def _format_docs(self, docs):
        """Format retrieved documents for prompt"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def create_qa_chain(
        self,
        k: int = 4
    ):
        """
        Create a QA chain with the vector store using LCEL
        
        Args:
            k: Number of documents to retrieve
        """
        self._retriever = self.vector_store_manager.as_retriever(
            search_kwargs={"k": k}
        )
        
        # Create chain using LCEL
        self.chain = (
            {"context": self._retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
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
        if self.chain is None or self._retriever is None:
            self.create_qa_chain(k=k)
        
        # Get answer from chain
        answer = self.chain.invoke(question)
        
        # Get source documents separately
        source_docs = self._retriever.invoke(question)
        
        return {
            'question': question,
            'answer': answer,
            'source_documents': source_docs
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
        # Recreate chain with new prompt
        self.chain = None
    
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
            model=self.llm_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_key=self.llm._client.api_key if hasattr(self.llm, '_client') else None
        )
        
        # Reset chain to use new LLM
        self.chain = None
    
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
            'chain_initialized': self.chain is not None
        }
