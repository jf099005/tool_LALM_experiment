def QA_pair_gen_prompt(n:int, audio_token, description, background_description, labels):

    TOOL_COUNT_HINTS = {
        1: "Single-tool questions. Examples: transcribe the whole audio, separate vocals only, clip a specific moment.",
        2: "Two-step questions. Examples: clip a segment then transcribe it; separate vocals then transcribe them; clip two different segments for comparison.",
        3: "Three-step chains. Examples: clip → separate → transcribe; or clip two segments (2 calls) then compare their transcripts (1 ASR call).",
        4: "Complex multi-step. Examples: clip → separate → ASR on one stem → ASR on another stem.",
    }

    tool_count_hint = TOOL_COUNT_HINTS[n]

    tool_definitions = '''The LALM has access to the following tools:

- **clipping**: Extract a time segment from an audio file.
Parameters: audio_id, audio_begin (HH:MM:SS.mmm), audio_end (HH:MM:SS.mmm)
Returns: a new clip_id and an <audio> token representing the clipped audio.

- **source_separation**: Separate an audio file into individual sound sources based on provided labels. Powered by SAM-Audio, which accepts any audio type (speech, music, environmental sounds, etc.) and separates sources according to the labels you specify.
Parameters:
- audio_id (string)
- audio_begin (HH:MM:SS.mmm)
- audio_end (HH:MM:SS.mmm)
- stems (array of strings): labels describing the sound sources to separate, e.g. ["speaker_1", "background_noise"], ["piano", "vocals"], ["engine_sound", "crowd"], ["dog_barking", "music"]. The labels should be grounded in the audio metadata provided.
Returns:
Separated audio stems for the specified segment. Each stem includes a semantic label, and an <audio> token representing the separated audio. Optional confidence scores may be provided.

- **asr**: Transcribe speech from audio to text.
Parameters: audio_id, audio_begin, audio_end
Returns: timestamped transcript with speaker_id and confidence.
'''


    audio_description = f"Description: {description}\n" if description else ""
    background_sound_description = f"Background sound description: {background_description}\n" if background_description else ""
    audio_labels = f"Labels: {labels}" if labels else ""
    prompt_template =\
f"""You are a data generation assistant. Your task is to create a question-answer pair that involves exactly {n} tool call(s) in its reasoning chain, based on the provided audio and its metadata.

---

## Audio Information
Audio ID: 0
Audio: {audio_token}
{audio_labels}{audio_description}{background_sound_description}
---

## Available Tools

{tool_definitions}

---

## Understanding Tool Roles

Tools are not only used to directly resolve a question. A tool call may serve one 
or more of the following roles in the reasoning chain:

- **Simplification**: Reduce audio complexity so the model can focus on the relevant 
  signal (e.g. separating vocals from background music before analysis, clipping a 
  relevant segment to avoid distraction from irrelevant content).

- **Augmentation**: Provide additional information that the model cannot reliably 
  perceive from raw audio alone (e.g. using ASR to obtain a reliable transcript, 
  using speaker separation to confirm speaker identity). This helps a weaker audio 
  model reason more accurately by grounding its analysis in tool-verified information.

- **Resolution**: Directly supply the data needed to answer the question.

---

## Your Task

Generate ONE multiple-choice question-answer pair, along with a complete reasoning chain showing how the answer is derived through tool use.
Your output consists of three parts:

1. **Question**: A multiple-choice question with 4 choices (A–D).
2. **Reasoning**: A step-by-step chain that invokes the necessary tools, interprets their results, identifies the correct choice.
3. **Answer**: The label (A, B, C, or D) of the correct choice.

All three parts must satisfy the requirements below.

### Question Requirements

1. The question MUST be grounded in the provided audio metadata — every aspect of the question must be supportable by the given description and labels. Do not introduce events, speakers, or content not evidenced in the metadata.
2. The question MUST have a single correct, verifiable answer. Do NOT generate open-ended, subjective, or opinion-based questions. The correct answer must be stateable as a short, unambiguous phrase or value.
3. The tool chain MUST meaningfully contribute to solving the question — each tool call must either simplify the audio into a more tractable form, or augment the model's understanding with information it could not reliably derive from the raw audio alone. The final answer may be reached by the model's own reasoning and perception after the tool chain has prepared the ground — it does not need to be mechanically read off from a tool result.
4. No tool call may be trivially skippable — if removing a tool call from the chain would not degrade the model's ability to answer (i.e. the raw audio and remaining tools are sufficient), that tool call must be removed or replaced.
5. The question should be natural and concise, as if asked by a real user who has not seen the metadata.


### Reasoning Requirements
1. Provide a step-by-step reasoning chain that shows how to arrive at the answer.
2. For each tool call, clearly state: why this tool is needed, what parameters to use, and what result is expected.
3. Each tool call must use the correct format defined in the Tool Invocation section below.
4. The reasoning must be consistent with the labels and description provided.
5. For each tool call, explicitly state its role: [Simplification], [Augmentation], [Resolution], or a combination.
6. The final reasoning step must identify the correct choice.

### Answer Requirements
1. Output only the label of the correct choice (A, B, C, or D).  Do not output the choice text.
2. The correct choice must be consistent with all tool results and the final reasoning conclusion. It need not be mechanically read off from a tool result directly.
3. Do NOT include reasoning or explanation in the answer field.

---

## Tool Invocation Format

When specifying a tool call in the reasoning chain, use the following format:

```json
{{
  "tool_call": {{
    "name": "<tool_name>",
    "request_id": "<unique_id>",
    "parameters": {{ ... }}
  }}
}}
```

Tool result placeholder format:
<tool_result>
[TOOL_RESULT_BEGIN]
request_id: <matching_id>
tool: <tool_name>
status: success
<expected result based on labels and description>
[TOOL_RESULT_END]
</tool_result>

---

## Output Format

You MUST output exactly three clearly separated blocks. Do not merge them.

### Block 1 — Question
[QUESTION]
<your question here>
[/QUESTION]

### Block 2 — Reasoning and Tool Use
[REASONING]
Step 1:
  Role: [Simplification | Augmentation | Resolution]
  Why this tool is needed: <...>
  Tool call: <tool_call json>
  Expected tool result: <tool_result block>
  Post-result interpretation: <how this changes the model's understanding>

(repeat for each tool call)

Final reasoning:
<derive the answer from all tool results>
Correct choice: <A | B | C | D> because <one-sentence justification>
[/REASONING]


### Block 3 — Answer
[ANSWER]
<A | B | C | D>
[/ANSWER]

---

## Self-Check Before Outputting

Before generating your final output, verify the following:

- [ ] All three blocks — [QUESTION], [REASONING], and [ANSWER] — are present and non-empty.
- [ ] Exactly {n} tool call(s) appear in the reasoning.
- [ ] The answer is uniquely determined — no ambiguity, no subjectivity.
- [ ] Every tool parameter (audio_id, timestamps, etc.) is consistent with the provided audio information.
- [ ] The answer matches what the tool results and reasoning actually conclude.
- [ ] The [QUESTION], [REASONING], and [ANSWER] blocks are fully separated with no cross-contamination.
- [ ] The question includes exactly 4 choices (A–D).
- [ ] Exactly one choice is correct; the remaining three are plausible distractors that cannot be eliminated without engaging with the audio or tool results.
- [ ] The final reasoning explicitly states which choice is correct and provides a justification.
- [ ] No distractor directly contradicts the metadata in an obvious way.
If any check fails, revise before outputting.
"""

    return prompt_template