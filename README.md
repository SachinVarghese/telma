# TELMA - Toolkit Evaluator for Language Model Agents

A toolkit evaluator for language model agents or assistants

## Define Tools

### Langchain Tools


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

### Huggingface Tools


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

## Assemble Toolkit


```python
from telma import ToolKit

tools = [tool1, tool2, tool3, tool4, tool5, tool6]
toolkit = ToolKit(tools=tools)
# toolkit.get_tools()
```

## Toolkit Comparison

### Design Evaluator


```python
from telma import SimpleEvaluator

evaluator = SimpleEvaluator()
```

### Assemble Toolkits


```python
from telma import ToolKit

tools = [tool1, tool2]
toolkit1 = ToolKit(tools=tools)
```


```python
from telma import ToolKit

tools = [tool2, tool3, tool4]
toolkit2 = ToolKit(tools=tools)
```

### Evaluate and Compare


```python
score1 = toolkit1.evaluate(evaluator=evaluator)
```


```python
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
