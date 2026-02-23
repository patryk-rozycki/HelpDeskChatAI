import hashlib
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
        
    def doc_id(self, doc: Document) -> str:
        content = f"{doc.metadata.get('source', '')}:{doc.page_content}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def comparison_new_document(self, documents: list[Document]) -> dict:
        existing_ids = set(self.store.get()["ids"])
        new_docs = []
        new_ids = []
        for doc in documents:
            doc_id = self.doc_id(doc)
            if doc_id not in existing_ids:
                new_docs.append(doc)
                new_ids.append(doc_id)
        if new_docs:
            self.store.add_documents(documents=new_docs, ids=new_ids)
        return {"added": len(new_docs), "skipped": len(documents) - len(new_docs)}
    
    def get_retriever(self) -> VectorStoreRetriever:
        return self.store.as_retriever(search_kwargs={"k": 8}, search_type="mmr", fetch_k=40)
