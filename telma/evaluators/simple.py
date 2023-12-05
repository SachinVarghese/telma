from .evaluator import Evaluator
from typing import List
from ..tool import Tool
from sentence_transformers import SentenceTransformer, util
import statistics


class SemanticDissimilarityEvaluator(Evaluator):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super()
        self.model = SentenceTransformer(model_name)

    def getCosineSimilarity(self, sentences: List[str]) -> float:
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings, embeddings)
        return cosine_scores.mean().item()

    def evaluate(self, tools: List[Tool]) -> float:
        name_cos_score = self.getCosineSimilarity([t.name for t in tools])
        desc_cos_score = self.getCosineSimilarity([t.description for t in tools])
        sign_cos_score = self.getCosineSimilarity(
            [str(t.signature_schema) for t in tools]
        )

        return 1 - statistics.mean([name_cos_score, desc_cos_score, sign_cos_score])
