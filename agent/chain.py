from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama
from agent.prompts import SYSTEM_PROMPT
from config import config
from rag.retriever import format_docs

class RagChain:
    def __init__(self, retriever: VectorStoreRetriever) -> None:
        self.prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
        
        self.llm = ChatOllama(
            model=config.llm_model,
            temperature=config.llm_temperature,
        )
        
        self.chain = (
            {"context": itemgetter("query") | retriever | format_docs,
             "query": itemgetter("query"),
             "history": itemgetter("history")
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def invoke(self, query: str, history: list[dict]) -> str:
        formatted_history = self._handle_format_history(history)
        return self.chain.invoke({"query": query, "history": formatted_history})
    
    def _handle_format_history(self, history: list[dict]) -> str:
        if not history:
            return ""
        lines = []
        for msg in history:
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)
