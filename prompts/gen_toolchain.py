guidelines = (
    "- The downstream model is weak and may struggle with ambiguous or implicit information.Prefer tools that produce clear, explicit, structured outputs (e.g., transcripts, labeled events, timestamps) over raw or abstract representations."
    # "- When in doubt, select more tools rather than fewer — the downstream model cannot infer missing information on its own."
    "- Avoid assuming the downstream model can reason across modalities or fill in gaps from incomplete tool outputs."
    "- Consider multi-step reasoning: a tool may not directly answer the question, but its output combined with the parameters used gives the downstream model a clearer signal to work with."
    "- Audio manipulation tools can be valuable even if they do not produce text — their outputs reduce ambiguity and narrow the downstream model's search space."
    # "- Audio manipulation tools (e.g., source separation, denoising, clipping) can be valuable even if they do not produce text — their outputs reduce ambiguity and narrow the downstream model's search space."
"- When assigning importance_score, use the full range of 1-10:"
  "\t- 9-10: The question almost certainly cannot be answered without this tool."
  "\t- 6-8: The tool significantly narrows down the correct answer."
  "\t- 3-5: The tool provides useful context but the question might still be answerable without it."
  "\t- 1-2: The tool offers only marginal supplementary information."

)

gen_tool_prompt_str = '''
You are an expert system for planning tool usage in audio question answering.

Your goal is to select a set of tools that helps another weaker language model answer correctly.
The downstream model has limited reasoning ability and cannot process raw audio directly.
It relies entirely on the structured text outputs you provide — the more explicit and 
pre-processed the information, the better it can perform.

You will be given:
- A question about an audio clip
- Multiple-choice options
- An audio input
- A list of available tools with descriptions

---

### Your task:

1. Analyze what information is required to answer the question:
   - Speech content (ASR)
   - Sound events (e.g., dog barking, car honking)
   - Speaker identity or count
   - Temporal information (timestamps)
   - Acoustic properties (pitch, loudness, emotion)
   - Music-related information

2. Select tools that can help answer the question, either directly or indirectly:
   - Directly: tools that extract explicit information (e.g., transcripts, timestamps, labeled sound events)
   - Indirectly: tools that isolate, enhance, or transform the audio so that the downstream model can more easily identify the relevant information.  
   
For example:
   - If the question asks about a specific sound event, consider whether separating or isolating that sound source (source separation) would make it easier to reason about its properties (e.g., timing, duration, intensity).
   - If the question focuses on a specific time range, consider clipping the audio to that segment so the downstream model can focus on the relevant portion without being distracted by irrelevant content.
   - If the audio contains heavy background noise that may obscure speech or sound events, consider denoising the audio first to improve the clarity of subsequent tool outputs or downstream model reasoning.
   - If the question involves speech content (e.g., what was said, word count, language spoken), ASR should be applied — and if the audio is noisy, consider combining denoising before ASR to improve transcription accuracy.
     
3. Avoid selecting tools that do not contribute useful information.

---

### Important Guidelines:

{guidelines}
---

### Context Passed to the Downstream Model:

The downstream model will receive the following information:
- The original question and multiple-choice options
- The tool calls you selected, including the tool name and parameters used
- The output returned by each tool

Therefore, you do not need to worry about whether the downstream model can trace 
where a piece of information comes from — it will have full visibility into both 
the tool parameters and their corresponding outputs.

Focus solely on selecting tools whose outputs provide information that is 
relevant and sufficient to answer the question.

---

### Question Information:

Question: {question}
Choices: {choices}
Audio: {audio_token}
Audio id: {audio_id}
Following are the descriptions of the tools you can use:
{tools_description}

### Output format:

[
    {{
        "tool": "ToolName",
        "role": "required" | "helpful",
        "importance_score": <integer from 1 to 10, where 10 means the question almost certainly cannot be answered correctly without this tool, and 1 means it provides only marginal supplementary information>,
        "reason": "Explain why this tool is useful and how its explicit output helps a weak downstream LLM — that cannot reason over raw audio — answer the question directly.",
        "parameters": {{"example_param": "value"}}
    }}
]
'''

output_note = '''Note: Always populate "parameters" with the specific configuration for the tool. Examples:
- Clipping:           {{"start_time": 2.0, "end_time": 6.0}}
- Source separation:  {{"source": "door"}}
- ASR:                {{"language": "en"}}
- Denoising:          {{}}  (no parameters required)
'''