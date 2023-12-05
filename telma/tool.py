from typing import Dict
from pydantic import BaseModel
from langchain.tools import BaseTool as Langchain_Tool
from openai.types import FunctionDefinition as OpenAI_Tool
from transformers import Tool as HuggingfaceHub_Tool


class Tool(BaseModel):
    name: str
    description: str
    signature_schema: Dict[str, object]

    @classmethod
    def from_langchain_tool(self, tool: Langchain_Tool):
        return self(
            name=tool.name,
            description=tool.description,
            signature_schema=tool.get_input_schema().schema(),
        )

    @classmethod
    def from_openai_function(self, tool: OpenAI_Tool):
        return self(
            name=tool.name,
            description=tool.description,
            signature_schema=tool.parameters,
        )

    @classmethod
    def from_huggingfaceHub(self, tool: HuggingfaceHub_Tool):
        return self(
            name=tool.name,
            description=tool.description,
            signature_schema={},
        )
