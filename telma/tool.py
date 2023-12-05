from typing import Dict
from pydantic import BaseModel
import sys

try:
    from langchain.tools import BaseTool as Langchain_Tool
except:
    Langchain_Tool = None

try:
    from openai.types import FunctionDefinition as OpenAI_Tool
except:
    OpenAI_Tool = None

try:
    from transformers import Tool as HuggingfaceHub_Tool
except:
    HuggingfaceHub_Tool = None

try:
    from llama_index.tools import BaseTool as LlamaIndex_Tool
except:
    LlamaIndex_Tool = None


class Tool(BaseModel):
    name: str
    description: str
    signature_schema: Dict[str, object]

    @classmethod
    def from_langchain_tool(self, tool: Langchain_Tool):
        if Langchain_Tool == None:
            print("Need `langchain` library installed", file=sys.stderr)
            raise ImportError
        return self(
            name=tool.name,
            description=tool.description,
            signature_schema=tool.get_input_schema().schema(),
        )

    @classmethod
    def from_openai_function(self, tool: OpenAI_Tool):
        if OpenAI_Tool == None:
            print("Need `openai` library installed", file=sys.stderr)
            raise ImportError
        return self(
            name=tool.name,
            description=tool.description,
            signature_schema=tool.parameters,
        )

    @classmethod
    def from_huggingfaceHub(self, tool: HuggingfaceHub_Tool):
        if HuggingfaceHub_Tool == None:
            print("Need `transformers` library installed", file=sys.stderr)
            raise ImportError
        return self(
            name=tool.name,
            description=tool.description,
            signature_schema={},
        )

    @classmethod
    def from_llamaIndex(self, tool: LlamaIndex_Tool):
        if LlamaIndex_Tool == None:
            print("Need `llama_index` library installed", file=sys.stderr)
            raise ImportError
        metadata = tool.metadata
        return self(
            name=metadata.name,
            description=metadata.description,
            signature_schema=metadata.fn_schema.schema(),
        )
