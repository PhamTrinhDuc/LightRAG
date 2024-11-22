from dataclasses import dataclass, field
from datetime import datetime
@dataclass
class Settings:
    # directory
    working_dir: str = field(
        default_factory= lambda: f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    #text chunk 
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 200
    tiktoken_model_name: str = "gpt-40-mini"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_token: int = 500

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimentions": 1536,
            "num_walks": 10,
            "walk_length": 10,
            "windown_size": 2,
            "iterations": 3,
            "random_seed": 3
        }
    )

    # config LLM
    