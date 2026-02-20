from langchain_classic.prompts import PromptTemplate
from langchain_classic.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama
from config import config
from langchain_core.documents import Document

prompt_template = """Use the context provided to answer 
the user's question below. If you do not know the answer 
based on the context provided, tell the user that you do 
not know the answer to their question based on the context
provided and that you are sorry.

context: {context}

question: {query}

answer: """

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class RagChain:
    def __init__(self, retriever: VectorStoreRetriever) -> None:
        self.prompt = PromptTemplate.from_template(prompt_template)
        
        self.llm = ChatOllama(
            model=config.llm_model,
            temperature=config.llm_temperature,
        )
        
        self.chain = (
            {"context": retriever | format_docs, "query": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def invoke(self, query: str) -> str:
        return self.chain.invoke(query)
    
