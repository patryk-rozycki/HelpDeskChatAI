from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever


def format_docs(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            formatted.append(f"[Source: {source}, page {page}\n{doc.page_content}]")
        return "\n\n".join(formatted)
    
class Retriever:
    def __init__(self, store: Chroma) -> None:
        self.store = store
    
    def get_retriever(self) -> VectorStoreRetriever:
        return self.store.as_retriever(search_kwargs={"k": 8}, 
                                       search_type="mmr", 
                                       fetch_k=40)

