from typing import List, Dict
import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

class DocumentProcessor:
    def __init__(self, persist_directory: str = "data/chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def process_pdf(self, pdf_path: str) -> List[str]:
        """Extract text from PDF and split into chunks."""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def create_vector_store(self, documents: List[str], metadata: List[Dict] = None):
        """Create or update vector store with document chunks."""
        if metadata is None:
            metadata = [{"source": f"document_{i}"} for i in range(len(documents))]
        
        vectorstore = Chroma.from_texts(
            texts=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            metadatas=metadata
        )
        vectorstore.persist()
        return vectorstore
    
    def get_vector_store(self):
        """Load existing vector store."""
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
    
    def process_and_index_document(self, pdf_path: str) -> None:
        """Process a PDF document and add it to the vector store."""
        chunks = self.process_pdf(pdf_path)
        metadata = [{"source": os.path.basename(pdf_path)} for _ in chunks]
        self.create_vector_store(chunks, metadata) 