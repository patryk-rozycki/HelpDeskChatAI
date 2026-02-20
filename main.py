from agent.ingest.loader import DocumentLoader
from agent.ingest.splitter import DocumentSplitter
from agent.retrieval.chain import RagChain
from agent.retrieval.vectorstore import VectorStore


loader = DocumentLoader()
splitter = DocumentSplitter()
vector_store = VectorStore()

docs = loader.load_pdfs()
all_splitter_documents = splitter.split_document(docs)
vector_store.reset_and_load(all_splitter_documents)
retriever = vector_store.get_retriever()

rag = RagChain(retriever)

query = input("Zadaj pytanie dla AI:")
response = rag.invoke(query)
print(response)

