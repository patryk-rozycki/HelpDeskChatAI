from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import config
from langchain_core.documents import Document

class DocumentSplitter:
    def __init__(self) -> None:
        self.splitter_chunk_size = config.splitter_chunk_size
        
        self.splitter_overlap = config.splitter_overlap
        
    def split_document(self, documents: list[Document]) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.splitter_chunk_size,
            chunk_overlap=self.splitter_overlap,
            add_start_index=True
        )
        return text_splitter.split_documents(documents)
