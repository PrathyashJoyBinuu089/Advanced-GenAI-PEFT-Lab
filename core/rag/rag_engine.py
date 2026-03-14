from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document

class RAGConfig(BaseModel):
    chunk_size: int = Field(1000, description="Size of text chunks")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    embedding_model: str = Field("text-embedding-3-small", description="OpenAI embedding model")
    llm_model: str = Field("gpt-4-turbo", description="OpenAI LLM model")
    vector_store_type: str = Field("faiss", description="Type of vector store (faiss, chroma)")
    persist_directory: Optional[str] = Field(None, description="Directory to persist vector store")

class RAGEngine:
    """
    Advanced RAG Engine for semantic search and context retrieval.
    """
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
        self.vector_store = None
        self.retrieval_chain = None
        self.llm = ChatOpenAI(model=self.config.llm_model, temperature=0)

    def load_and_index(self, file_path: str):
        """Loads documents from a file and indexes them into the vector store."""
        print(f"Loading document from {file_path}...")
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
            
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        print(f"Split document into {len(texts)} chunks.")

        if self.config.vector_store_type.lower() == "faiss":
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
        else:
            self.vector_store = Chroma.from_documents(
                texts, 
                self.embeddings, 
                persist_directory=self.config.persist_directory
            )
        
        self.setup_qa_chain()

    def setup_qa_chain(self):
        """Initializes the RetrievalQA chain."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Index documents first.")
            
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        self.retrieval_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

    def query(self, question: str):
        """Executes a RAG query."""
        if not self.retrieval_chain:
            # Fallback if no docs indexed
            return {"result": "No documents indexed. Please index data first.", "source_documents": []}
            
        response = self.retrieval_chain.invoke({"query": question})
        return response

if __name__ == "__main__":
    # Example usage
    # config = RAGConfig()
    # engine = RAGEngine(config)
    print("RAG Engine logic ready.")
