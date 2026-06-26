from __future__ import annotations
from pathlib import Path
import sys
from tools import generate_tool_descriptions


repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'sam-audio'))

from .gen_toolchain import gen_tool_prompt_str, guidelines
from .audio_QA import MCQ_system_prompt, MCQ_user_turn_template, workflow, only_answer_workflow
from .tool_chain_result_read import read_tool_chain


def fill_template(template_str, **kwargs):
    return template_str.format(**kwargs)

def gen_tool_prompt(question, options, audio_token, tool_classes):
    tools_description = generate_tool_descriptions(tool_classes)
    return gen_tool_prompt_str.format(
        question=question,
        options=options,
        audio_token=audio_token,
        tools=tools_description,
        guidelines=guidelines
    )

def QA_prompt(question, options,  audio_token, final_round = False, tool_results = None, tools_list = None, tools_description = None):
    if not final_round:
        MCQ_system_prompt_filled = MCQ_system_prompt.format(
            system_prompt="You are an expert in audio analysis and question answering.",
            tools_description=tools_description if tools_description is not None else generate_tool_descriptions(tools_list),
            workflow=workflow
        )
    else:
        MCQ_system_prompt_filled = only_answer_workflow.format(
            system_prompt="You are an expert in audio analysis and question answering.",
            tools_description=tools_description if tools_description is not None else generate_tool_descriptions(tools_list)
        )

    if tool_results is None:
        tool_results = "No tools have been invoked yet."

    MCQ_user_turn_filled = MCQ_user_turn_template.format(
        question=question,
        options=options,
        audio=audio_token,
        tool_results=tool_results
    )

    return MCQ_system_prompt_filled, MCQ_user_turn_filled