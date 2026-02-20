from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from config import config

class VectorStore:
    def __init__(self) -> None:
        self.embeddings = OllamaEmbeddings(model=config.embedding_model)
        
        self.store = Chroma(
            collection_name=config.chroma_collection_name,
            embedding_function=self.embeddings,
            persist_directory=config.chroma_persist_directory,
        )
    
    def reset_and_load(self, all_splitter_documents: list[Document]) -> list[str]:
        self.store.reset_collection()
        return self.store.add_documents(documents=all_splitter_documents)
    
    def get_retriever(self) -> VectorStoreRetriever:
        return self.store.as_retriever()
