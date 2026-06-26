
import json
import os
import re

TOOL_RESULT_TEMPLATE = '''
### Tool: {tool}
Input Parameters: {parameters}
Output: {output}
'''

def extract_target_tool_results(parameters, text_output, audio_token):
    target = parameters.get('target_description')
    if target is None:
        raise ValueError("Expected 'target' parameter not found in tool input.")
    return f"{target}-isolated auxiliary audio: {audio_token}"

def remove_target_tool_results(parameters, text_output, audio_token):
    target = parameters.get('target_description')
    if target is None:
        raise ValueError("Expected 'target' parameter not found in tool input.")
    return f"auxiliary audio with {target} removed: {audio_token}"

def asr_tool_results(parameters, text_output, audio_token):
    transcription = text_output
    if transcription is None:
        raise ValueError("Expected 'transcription' parameter not found in tool input.")
    return f"transcription of the audio: {transcription}"

def human_voice_enhance_tool_results(parameters, text_output, audio_token):
    return f"human voice enhanced audio: {audio_token}"

def pitch_shift_tool_results(parameters, text_output, audio_token):
    return f"pitch-shifted audio: {audio_token}"

READ_TOOL_RESULTS_FUNCTIONS = {
    'extract_target': extract_target_tool_results,
    'remove_target': remove_target_tool_results,
    'asr': asr_tool_results,
    'human_voice_enhance': human_voice_enhance_tool_results,
    'pitch_shift': pitch_shift_tool_results,
    'time_stretch': lambda parameters, text_output, audio_token: f"time-stretched audio: {audio_token}"
}

def read_tool_chain(tool_chain_folder_path, audio_token, indexing = False, specify_tool = None, skip_illegal = False, index_format = 'order'):
    #index_format can be 'name' or 'order'
    ordered = []

    for idx, filename in enumerate(os.listdir(tool_chain_folder_path)):
        name = os.path.splitext(filename)[0]
        if index_format == 'order':
            m = re.search(r'tool_input_(\d+)\.json', filename)
        elif index_format == 'name':
            m = re.search(r'tool_inputs_.*\.json', filename)
        else:
            raise ValueError(f"Invalid index_format: {index_format}. Expected 'name' or 'order'.")

        if m:
            if index_format == 'order':
                tool_order = int(m.group(1))
            else:
                tool_order = 0

            with open(os.path.join(tool_chain_folder_path, filename), 'r') as f:
                tool_input = json.load(f)
            tool = tool_input['tool']
            tool_result_path = name.replace('inputs', 'result')
            tool_result_path = name.replace('input', 'result')
            text_output = None
            output_audio_path = None

            text_output_path = os.path.join(tool_chain_folder_path, tool_result_path + '.txt')
            output_audio_path = os.path.join(tool_chain_folder_path, tool_result_path + '.wav')
            
            assert not (os.path.exists(text_output_path) and os.path.exists(output_audio_path)), f"Both text and audio output exist for tool {tool} with paths: {text_output_path} and {output_audio_path}. This is unexpected."

            if os.path.exists(text_output_path):
                output_audio_path = None
                with open(os.path.join(tool_chain_folder_path, tool_result_path + '.txt'), 'r') as f:
                    text_output = f.read()
            elif os.path.exists(output_audio_path):
                text_output = None
                assert os.path.exists(output_audio_path), f"Expected audio output not found: {output_audio_path}"
            
            elif not skip_illegal:
                raise FileNotFoundError(f"No output found for tool {tool} with expected paths: {text_output_path} or {output_audio_path}")
            else:
                continue
            parameters = tool_input['parameters']
            parameters.pop('audio_path', None)  # Remove the input audio path from parameters

            assert tool in READ_TOOL_RESULTS_FUNCTIONS, f"No result extraction function defined for tool: {tool}"
            tool_output = READ_TOOL_RESULTS_FUNCTIONS[tool](parameters, text_output, audio_token)
            if indexing:
                tool_output = f"{idx}. {tool_output}\n"
    
            if specify_tool is None or tool == specify_tool:
                ordered.append((tool_order, tool_output, output_audio_path))
            else:
                assert not indexing
    if len(ordered) == 0:
        print(ordered)
        print(f"Files in tool chain folder: {os.listdir(tool_chain_folder_path)}")
        raise ValueError(f"No tool results found in folder: {tool_chain_folder_path} with specify_tool={specify_tool}.")
    ordered.sort(key=lambda x: x[0])
    # tool_chain_results = [r for _, r, _ in ordered]
    tool_chain_output = "".join([r for _, r, _ in ordered])
    tool_audio_path = [p for _, _, p in ordered if p is not None]

    return tool_chain_output, tool_audio_path