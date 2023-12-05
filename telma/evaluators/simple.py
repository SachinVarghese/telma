from .evaluator import Evaluator
from typing import List
from ..tool import Tool
from sentence_transformers import SentenceTransformer, util


class SemanticDissimilarityEvaluator(Evaluator):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super()
        self.model = SentenceTransformer(model_name)

    def evaluate(self, tools: List[Tool]) -> float:
        embeddings = self.model.encode(
            [t.model_dump_json() for t in tools], convert_to_tensor=True
        )
        cosine_scores = util.cos_sim(embeddings, embeddings)
        return 1 - cosine_scores.mean().item()
