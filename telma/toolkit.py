from typing import List
from .evaluators import Evaluator
from .tool import Tool
from langchain.tools import BaseTool as Langchain_Tool
from openai.types import FunctionDefinition as OpenAI_Tool
from transformers import Tool as HuggingfaceHub_Tool


class ToolKit:
    def __init__(self, tools: List[Tool]):
        self.tool_list = tools

    def get_tools(self):
        return self.tool_list

    def evaluate(self, evaluator: Evaluator) -> float:
        return evaluator.evaluate(self.tool_list)

    @classmethod
    def from_langchain_tools(self, tools: List[Langchain_Tool]):
        toolset = [
            Tool(
                name=t.name,
                description=t.description,
                signature_schema=t.get_input_schema().schema(),
            )
            for t in tools
        ]
        return self(tools=toolset)

    @classmethod
    def from_openai_functions(self, tools: List[OpenAI_Tool]):
        toolset = [
            Tool(name=t.name, description=t.description, signature_schema=t.parameters)
            for t in tools
        ]
        return self(tools=toolset)

    @classmethod
    def from_huggingfaceHub(self, tools: List[HuggingfaceHub_Tool]):
        toolset = [Tool(name=t.name, description=t.description) for t in tools]
        return self(tools=toolset)
