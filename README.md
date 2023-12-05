# TELMA - Toolkit Evaluator for Language Model Agents

TELMA is a toolkit evaluator for language model agents or assistants. 

## Introduction

AI assistants or agents can be built by leveraging an agentic language model behaviour. Agentic behaviour is the ability to use external tools in order to solve tasks. For this a language model is prompted with a set of tool definitions and instructions on how to use these tools to complete a certain task.

The ability of the language model to utilize these tools efficiently depends not only on the tool definitions but also what tools are used together in a language model prompt. This project aims at evaluating and comparing different toolkits i.e combinations of such tool definitions.

## Usage

TELMA provides interfaces to define language model agent tools, assemble them as toolkits and run evaluations on these toolkits based on any defined heuristic. TELMA provides some evaluation heuristics out of the box that can be used to score and compare toolkits or extended to build custom evaluators. So the main usage steps are

- Defines language model agent tools
- Define toolkits as combination of tools
- Evaluate Individual toolkit on a heuristic
- Compare toolkits to choose the best fit

## Define Tools

Tools can be defined in TELMA in multiple ways. There are many frameworks/projects that help build language model based agents/ assistants and hosts a set of tools. TELMA aims to integrate with most of such projects to define tools and compare toolkits.
- Native definition (see schema definition for details)
- From Langchain Hub tools
- From Open AI functions
- From Huggingface Hub Tools
- From LlamaIndex Module Tools


```python
from telma import Tool
from pprint import pprint

pprint(Tool.model_json_schema())
```

    {'properties': {'description': {'title': 'Description', 'type': 'string'},
                    'name': {'title': 'Name', 'type': 'string'},
                    'signature_schema': {'title': 'Signature Schema',
                                         'type': 'object'}},
     'required': ['name', 'description', 'signature_schema'],
     'title': 'Tool',
     'type': 'object'}


### Native Tool Definition

Lets define some tools


```python
from telma import Tool

tool0 = Tool(
    name="example tool",
    description="example tool description",
    signature_schema={"type": "string"},
)
```

### Langchain Tools

Lets define some tools from Langchain Hub


```python
# !pip install -U duckduckgo-search
from langchain.tools import DuckDuckGoSearchRun

tool1 = DuckDuckGoSearchRun()
```


```python
from langchain.tools import BraveSearch

tool2 = BraveSearch.from_api_key(api_key="xxx", search_kwargs={"count": 3})
```


```python
from telma import Tool

tool1 = Tool.from_langchain_tool(tool1)
tool2 = Tool.from_langchain_tool(tool2)
```

### Open AI Functions

Now lets define some tools from Open AI function definitions


```python
from openai.types import FunctionDefinition

tool3_params = {
    "type": "object",
    "properties": {
        "to": {"type": "string"},
        "body": {"type": "string"},
    },
    "additionalProperties": "false",
}
tool3 = FunctionDefinition(
    name="send_email", parameters=tool3_params, description="Send an email"
)
```


```python
from openai.types import FunctionDefinition

tool4_params = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "unit": {"enum": ["celsius", "fahrenheit", "kelvin"]},
    },
    "additionalProperties": "false",
}


tool4 = FunctionDefinition(
    name="get_current_weather",
    parameters=tool4_params,
    description="Find out about weather",
)
```


```python
from telma import Tool

tool3 = Tool.from_openai_function(tool3)
tool4 = Tool.from_openai_function(tool4)
```

### Huggingface Hub Tools

Now lets define some tools from Huggingface Hub


```python
# !pip install diffusers accelerate
from transformers import load_tool

tool5 = load_tool("text-to-speech")
tool6 = load_tool("huggingface-tools/text-to-image")
```


```python
from telma import Tool

tool5 = Tool.from_huggingfaceHub(tool5)
tool6 = Tool.from_huggingfaceHub(tool6)
```

### Llama Index

Now lets define some tools from Llama Index


```python
from llama_index.tools import QueryEngineTool, ToolMetadata

tool7 = QueryEngineTool(
    query_engine=None,
    metadata=ToolMetadata(
        name="lyft_10k",
        description=(
            "Provides information about Lyft financials for year 2021. "
            "Use a detailed plain text question as input to the tool."
        ),
    ),
)
```


```python
from telma import Tool

tool7 = Tool.from_llamaIndex(tool7)
```

## Toolkit Assembly

Once you have defined a set of tools, a toolkit can be created including as many tools as follows 


```python
from telma import ToolKit

tools = [tool0, tool1, tool2, tool3, tool4, tool5, tool6, tool7]
toolkit = ToolKit(tools=tools)

# toolkit.get_tools()
```

### Assemble Two Toolkits for Comparison

For our example lets define two different toolkits, `Toolkit 1` with two similar search tools and `Toolkit 2` with three varied tools. 


```python
from telma import ToolKit

toolkit1 = ToolKit(tools=[tool1, tool2])
toolkit2 = ToolKit(tools=[tool2, tool3, tool4])
```

Logically a language model should have difficulty choosing between tools in Toolkit 1 compared to Toolkit 2 due to the similarity of tools available and hence would be less efficient in choosing the right tool for a job. Now lets evaluate them.

## Toolkit Evaluation and Comparison

### Define/Design Evaluation Heuristic

For this example we utilise the out of the box semantic similary evaluator to score a toolkit. The idea here is that the variety in tool definitions makes the job of the language model easier to choose between the tools for different purposes.

The `Evaluator` class in TELMA can be extended to create custom evaluation heuristics.


```python
from telma import SemanticDissimilarityEvaluator

evaluator = SemanticDissimilarityEvaluator()
```

### Evaluate and compare toolkits

Lets compute evalaution score based on our defined heuristic.


```python
score1 = toolkit1.evaluate(evaluator=evaluator)
score2 = toolkit2.evaluate(evaluator=evaluator)
```


```python
if score1 > score2:
    print("Toolkit 1 is better that Toolkit 2!")
else:
    print("Toolkit 2 is better that Toolkit 1!")
```

    Toolkit 2 is better that Toolkit 1!


## References

- [Langchain Hub Tools](https://python.langchain.com/docs/modules/agents/tools/custom_tools)
- [Open AI functions](https://platform.openai.com/docs/guides/function-calling)
- [Huggingface Hub Tools](https://huggingface.co/docs/transformers/custom_tools)
- [LlamaIndex Module Tools](https://docs.llamaindex.ai/en/stable/optimizing/agentic_strategies/agentic_strategies.html)
- Tool Schema Definition - [JSON Schema](https://json-schema.org/understanding-json-schema/)
- Tool Comparison - [Semantic Textual Similarity](https://www.sbert.net/docs/usage/semantic_textual_similarity.html)
