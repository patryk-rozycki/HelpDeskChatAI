from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama
from config import config

prompt_template = """You are a professional Product & Policy Consultant.
    Your role is to help customers with questions about products, shipping, 
    returns, and company policies.

    Rules:
    1. Answer ONLY based on the context below. Be helpful, professional, and concise.
    2. If the context contains the answer, provide it and cite the source: [Source: filename, page N].
    3. If the context does NOT contain the answer, say: "I don't have information about this in our documents. Please contact   support."
    4. Never guess or invent facts about products, prices, or policies.
    5. Respond in the same language as the user's question.

    Context:
    {context}

    Question: {query}

    Answer: """

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
    
