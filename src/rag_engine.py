from typing import List, Dict, Optional
from .document_processor import DocumentProcessor
from .llm_interface import LLMInterface

class RAGEngine:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.llm_interface = LLMInterface()
    
    def process_document(self, pdf_path: str) -> None:
        """Process and index a new document."""
        self.document_processor.process_and_index_document(pdf_path)
    
    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context from the vector store."""
        vectorstore = self.document_processor.get_vector_store()
        docs = vectorstore.similarity_search(query, k=k)
        return "\n\n".join(doc.page_content for doc in docs)
    
    def generate_answer(self, question: str) -> str:
        """Generate an answer using RAG."""
        # First check if the question is relevant
        if not self.llm_interface.is_relevant_question(question):
            return "I apologize, but I can only answer questions related to Indian Labour and Consumer Court law. Your question appears to be outside this domain."
        
        # Get relevant context
        context = self.get_relevant_context(question)
        
        # Generate response
        response = self.llm_interface.generate_response(question, context)
        return response
    
    def clear_documents(self) -> None:
        """Clear all indexed documents."""
        import shutil
        import os
        if os.path.exists(self.document_processor.persist_directory):
            shutil.rmtree(self.document_processor.persist_directory) 