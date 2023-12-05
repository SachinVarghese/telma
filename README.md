# TELMA - Toolkit Evaluator for Language Model Agents

TELMA is a toolkit evaluator for language model agents or assistants. 

## Intoduction

AI assistants or agents can be built by leveraging an agentic language model behaviour. Agentic behaviour is the ability to use external tools in order to solve tasks. For this a language model is prompted with a set of tool definitions and instructions on how to use these tools to complete a certain task.

The ability of the language model to utilize these tools efficiently depends not only on the tool definitions but also what tools are used together in a language model prompt. This project aims at evaluating and comparing different toolkits i.e combinations of such tool definitions.

## Usage

TELMA provides interfaces to define language model agent tools, assemble them as toolkits and score different based on any defined heuristic. TELMA provides some evaluation heuristics out of the box that can be used to compare toolkits or extended to build custom evaluators. Main usage steps are

- Defines agent tools
- Define toolkits as combination of tools
- Evaluate Individual toolkit on a heuristic
- Compare toolkits to choose the best fit

### Define Tools

Tools can be defined in TELMA in multiple ways. There are many frameworks/projects that help build language model based agents/ assistants and hosts a set of tools. TELMA aims to integrate with most of such projects to define tools and compare toolkits.
- Native definition (see schema definition for details)
- From Langchain Hub tools
- From Open AI functions
- From Huggingface Hub Tools


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
```


```python
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

### Example Toolkit Assembly

Once you have defined a set of tools, a toolkit can be created as follows


```python
from telma import ToolKit

tools = [tool1, tool2, tool3, tool4, tool5, tool6]
toolkit = ToolKit(tools=tools)
# toolkit.get_tools()
```

## Toolkit Evaluation and Comparison

### Assemble Two Toolkits

For our example lets define two different toolkits, `Toolkit 1` with two similar search tools and `Toolkit 2` with three varied tools. 


```python
from telma import ToolKit

toolkit1 = ToolKit(tools=[tool1, tool2])
toolkit2 = ToolKit(tools=[tool2, tool3, tool4])
```

Logically a language model should have difficulty choosing between tools in Toolkit 1 compared to Toolkit 2 due to the similarity of tools available and hence would be less efficient in choosing the right tool for a job. Now lets evaluate them.

### Define/Design Evaluation Heuristic

For this example we utilise the out of the box semantic similary evaluator to score a toolkit. The idea here is that the variety in tool definitions makes the job of the language model easier to choose between the tools for different purposes.

The `Evaluator` class in TELMA can be extended to create custom evaluation heuristics.


```python
from telma import SemanticDissimilarityEvaluator

evaluator = SemanticDissimilarityEvaluator()
```

### Evaluate and Compare

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

- Tool Schema Definition - [JSON Schema](https://json-schema.org/understanding-json-schema/)
- Tool Comparison - [Semantic Textual Similarity](https://www.sbert.net/docs/usage/semantic_textual_similarity.html)
