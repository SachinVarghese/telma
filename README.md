# TELMA

### Toolkit Evaluator for Language Model Agents

> Equip your language model agents or AI assistants with the best set of tool definitions.

### Abstract

The ability of the agentic language model to efficiently utilize a set of tools depends not only on the individual tool definitions but also on what tools are grouped in a context. The agentic behaviour using language models can be hugely enhanced by choosing the right set of tool definitions in a context.

TELMA is a toolkit evaluator for language model agents or AI assistants. TELMA aims to assist with the evaluation and comparison of different combinations of such tool definitions to improve the agentic behaviour.

Source code available on GitHub: [https://github.com/SachinVarghese/telma](https://github.com/SachinVarghese/telma)

### Contents

* [Background](#background)
* [Usage](#usage)
* [Tool definitions](#tool-definitions)
* [Toolkit assembly](#toolkit-assembly)
* [Toolkit evaluation and comparison](#toolkit-evaluation-and-comparison)
    * [Define evaluation heuristic](#define-evaluation-heuristic)    
    * [Evaluate and compare toolkits](#evaluate-and-compare-toolkits)  
* [References](#references)

## Background

AI assistants or agents can be built by leveraging an agentic language model behavior. Agentic behavior is the ability of a language model to plan the next steps as actions and utilize external tools to solve tasks. For this, a language model is prompted with a set of tool definitions and instructions on how to use these tools in order to complete different types of tasks. A combination of such tools in a language model context can be referred to as an agent toolkit.

## Usage

TELMA provides interfaces to define language model agent tools, assemble them as toolkits, and run evaluations on these toolkits based on any defined heuristic. TELMA provides some evaluation heuristics out of the box that can be used to score and compare toolkits or extended to build custom evaluators. So the main usage steps are

- Defines tools for a language model agent
- Define toolkits as a group of such tools
- Evaluate toolkits on any user-defined heuristic
- Compare toolkits to choose the best fit for an agent


```python
from telma import Tool, ToolKit

tool = Tool(
    name="python code interpreter",
    description="this tool interprets python code and responds with the log results",
    signature_schema={"type": "string"},
)

toolkit = ToolKit(tools=[tool])
```


```python
from telma.evaluators import SemanticDissimilarityEvaluator

toolkit.evaluate(evaluator=SemanticDissimilarityEvaluator())
```




    -3.9736429924275285e-08



## Tool definitions

Many frameworks/projects help build language model-based agents or AI assistants like Langchain, Huggingface, etc. TELMA aims to integrate with most such projects to define tools and compare toolkits. Agentic Tools can be defined in TELMA in multiple ways as follows,
- Native definition (see schema definition for details)and host a set of tools
- From Langchain Hub tools
- From Open AI functions
- From Huggingface Hub Tools
- From LlamaIndex Module Tools

### Native tool definition


```python
tool0 = Tool(
    name="Google search",
    description="This tool helps to retrieve information from Google search results",
    signature_schema={"type": "string"},
)
```

### Langchain Tools


```python
# !pip install -U langchain duckduckgo-search
from langchain.tools import DuckDuckGoSearchRun, BraveSearch
```


```python
tool1 = Tool.from_langchain_tool(DuckDuckGoSearchRun())
```


```python
tool2 = Tool.from_langchain_tool(
    BraveSearch.from_api_key(api_key="xxx", search_kwargs={"count": 3})
)
```

### Open AI Functions


```python
# !pip install -U openai
from openai.types import FunctionDefinition
```


```python
tool3 = Tool.from_openai_function(
    FunctionDefinition(
        name="send_email",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "body": {"type": "string"},
            },
            "additionalProperties": "false",
        },
        description="Send an email",
    )
)
```


```python
tool4 = Tool.from_openai_function(
    FunctionDefinition(
        name="get_current_weather",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"enum": ["celsius", "fahrenheit", "kelvin"]},
            },
            "additionalProperties": "false",
        },
        description="Find out about weather",
    )
)
```

### Huggingface Hub Tools


```python
# !pip install -U transformers diffusers accelerate
from transformers import load_tool
```


```python
tool5 = Tool.from_huggingfaceHub(load_tool("text-to-speech"))
```


```python
tool6 = Tool.from_huggingfaceHub(load_tool("huggingface-tools/text-to-image"))
```

### Llama Index


```python
# !pip install -U llama_index
from llama_index.tools import QueryEngineTool, ToolMetadata
```


```python
tool7 = Tool.from_llamaIndex(
    QueryEngineTool(
        query_engine=None,
        metadata=ToolMetadata(
            name="lyft_2021",
            description=(
                "Provides information about Lyft financials for year 2021."
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    )
)
```



## Toolkit assembly

Most language model agents work with one or more tools in a context. Once you have defined a set of tools, a toolkit can be assembled follows:


```python
from telma import ToolKit

tools = [tool0, tool1, tool2, tool3, tool4, tool5, tool6, tool7]
toolkit = ToolKit(tools=tools)
```

### Assemble 2 toolkits for comparison

To demonstrate toolkit comparison, let's define two different toolkits, 
- `Toolkit 1` with three similar search tools [Google Search, DuckDuckGo Search and Brave Search] 
- `Toolkit 2` with three different tools [send_email, get_current_weather, text-to-speech]


```python
toolkit1 = ToolKit(tools=[tool0, tool1, tool2])
toolkit2 = ToolKit(tools=[tool3, tool4, tool5])
```

## Toolkit evaluation and comparison

In theory, an agentic behaviour powered by a language model should have more difficulty differentiating and choosing between the tools in `Toolkit 1` compared to `Toolkit 2` due to the similarity of tool definitions. So the agent would be less efficient in selecting the right tool when using `Toolkit 1` especially if the language model is less powerful. 

For such reason, it is extremely important to evaluate toolkits to understand the efficiency of agentic behaviour using language models.

### Define Evaluation Heuristic

For the toolkit evaluation, we utilize the out-of-the-box semantic dissimilarity evaluator. The heuristic with this evaluation is that the variance in tool naming and descriptions makes it easier for a language model to choose between the tools for different requirements. Such a  criteria be extremely useful with smaller language models(fewer parameters) that are less powerful. 


```python
from telma import SemanticDissimilarityEvaluator

evaluator = SemanticDissimilarityEvaluator()
```

> Note: The `Evaluator` class in TELMA can also be extended to create custom evaluation heuristics. TELMA users are encouraged to create custom toolkit evaluators to meet their agent requirements.

### Evaluate and compare toolkits

Now let's compute the evaluation score based on our defined heuristic.


```python
score1 = toolkit1.evaluate(evaluator=evaluator)
score2 = toolkit2.evaluate(evaluator=evaluator)
```


```python
if score1 > score2:
    print("Toolkit 1 is better suited for my agent!")
else:
    print("Toolkit 2 is better suited for my agent!")
```

    Toolkit 2 is better suited for my agent!


Hope this project helps build better language model agents!

## References

- [Langchain Hub Tools](https://python.langchain.com/docs/modules/agents/tools/custom_tools)
- [Open AI functions](https://platform.openai.com/docs/guides/function-calling)
- [Huggingface Hub Tools](https://huggingface.co/docs/transformers/custom_tools)
- [LlamaIndex Module Tools](https://docs.llamaindex.ai/en/stable/optimizing/agentic_strategies/agentic_strategies.html)
- Tool Schema Definition - [JSON Schema](https://json-schema.org/understanding-json-schema/)
- Tool Comparison - [Semantic Textual Similarity](https://www.sbert.net/docs/usage/semantic_textual_similarity.html)
