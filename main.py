from agent.ingest.loader import DocumentLoader
from agent.ingest.splitter import DocumentSplitter
from agent.retrieval.chain import RagChain
from agent.retrieval.vectorstore import VectorStore


loader = DocumentLoader()
splitter = DocumentSplitter()
vector_store = VectorStore()

docs = loader.load_pdfs()
all_splitter_documents = splitter.split_document(docs)
# vector_store.reset_and_load(all_splitter_documents)
handle_vector_documents = vector_store.comparison_new_document(all_splitter_documents)
print(handle_vector_documents)
retriever = vector_store.get_retriever()

rag = RagChain(retriever)

while True:
    query = input("Zadaj pytanie dla AI, lub wpisz exit: ")
    if query.lower() == "exit":
        print("Do zobaczenia!")
        break
    response = rag.invoke(query)
    print(response)

