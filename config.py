from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # LLM
    llm_model: str = "llama3.1"
    llm_temperature: float = 0.0
    embedding_model: str = "mxbai-embed-large:latest"

    # Vector_store
    chroma_collection_name: str = "example_collection"
    chroma_persist_directory: str = "./data/chroma"

    # Loader
    source_dir: str = "data/raw"
    
    #Splitter
    splitter_chunk_size: int = 1000
    splitter_overlap: int = 200

config = Config()
