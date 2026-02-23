from ingest.loader import DocumentLoader
from ingest.splitter import DocumentSplitter


def run_ingest():
    loader = DocumentLoader()
    splitter = DocumentSplitter()
    docs = loader.load_pdfs()
    
    return splitter.split_document(docs)
