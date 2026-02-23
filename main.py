from agent.chain import RagChain
from ingest.pipeline import run_ingest
from rag.retriever import Retriever
from rag.vectorstore import VectorStore

vector_store = VectorStore()

handle_vector_documents = vector_store.comparison_new_document(run_ingest())
retriever = Retriever(vector_store.store)
rag = RagChain(retriever.get_retriever())

# Add addicional db for history per user
history = []

while True:
    query = input("Zadaj pytanie dla AI, lub wpisz exit: ")
    if query.lower() == "exit":
        print("Do zobaczenia!")
        break
    response = rag.invoke(query, history)
    history.append({"role": "user", "content": query })
    history.append({"role": "assistent", "content": response})
    print(response)
