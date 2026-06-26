audio_element_template = '''
audio id: {audio_id}
audio: {audio_token}
'''

def gen_audio_element(audio_id, audio_token):
    return audio_element_template.format(
        audio_id=audio_id,
        audio_token=audio_token
    )