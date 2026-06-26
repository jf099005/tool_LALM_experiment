MCQ_system_prompt = '''
{system_prompt}

You are given a question about an audio clip along with multiple-choice options.

{tools_description}

{workflow}
'''

workflow = '''
## Workflow
You have access to a set of tools that can extract information from the audio.
You operate in a multi-turn loop. At each turn, you receive the original question and the results of any tools invoked so far. You then choose one of two actions:

**Action A — Invoke tool(s)**
If the current information is insufficient to answer confidently, request one or more tools.
Output a JSON list:

[
  {{
    "tool": "ToolName",
    "role": "required" | "helpful",
    "importance_score": <integer 1–10>,
    "reason": "Explain why this tool is needed given what you already know, and how its output helps answer the question.",
    "parameters": {{"example_param": "value"}}
  }}
]

**Action B — Answer**
If you have enough information to answer confidently, output:

{{
  "answer": "<option>",
  "reason": "<explanation based on tool results and/or prior knowledge>"
}}
'''

only_answer_workflow = '''
### Your Task
Based on all previously collected information, answer the question.
'''

# 每輪 user message 的 template
MCQ_user_turn_template = '''
"Focus on the audio clips and answer the question."
## Question
{question}

## Audio
{audio}

## Options
{options}

## Tool Results So Far
Before the current turn, you have invoked some tools and received their results. Here is a summary of the information you have so far:
{tool_results}
'''