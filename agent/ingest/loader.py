from langchain_classic.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from config import config

class DocumentLoader:
    def __init__(self, source_dir: str | None = None) -> None:
        self.source_dir = source_dir or config.source_dir
    
    def load_pdfs(self) -> list[Document]:
        loader = PyPDFDirectoryLoader(self.source_dir)
        return loader.load()
