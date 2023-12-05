from typing import List
from ..tool import Tool


class Evaluator:
    def evaluate(self, tools: List[Tool]) -> float:
        raise NotImplementedError
