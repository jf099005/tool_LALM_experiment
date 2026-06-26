import os
import io
import time
import json
# from tqdm import tqdm
import soundfile as sf
import librosa
from google import genai
from google.genai import types
from argparse import ArgumentParser, Namespace
import random
import copy

from concurrent.futures import ThreadPoolExecutor, as_completed

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

TARGET_ORG = "Multi-Audio-Grounding"
MODEL_ID = "gemini-3-pro-preview"
TARGET_SR = 16000

def load_audio(audio_path):
    waveform, sr = librosa.load(audio_path, sr=None)  # 保留原始 sampling rate
    return {
        "array": waveform,
        "sampling_rate": sr
    }

def numpy_to_wav_bytes(audio_data):
    if audio_data is None:
        return None

    waveform = audio_data['array']
    sr = audio_data['sampling_rate']

    # Resample to target sampling rate if necessary
    if sr != TARGET_SR:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # Convert to WAV bytes
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sr, format='WAV')
    buffer.seek(0)
    return buffer.read()

def build_content(instruction, audio_token, audio_path):
    parts = []

    instruction_L, instruction_R = instruction.split(audio_token)

    audio_data = load_audio(audio_path)
    wav_bytes = numpy_to_wav_bytes(audio_data)
    
    # text before audio
    parts.append(types.Part.from_text(text=instruction_L))

    # audio
    if wav_bytes:
        parts.append(types.Part.from_bytes(
            data=wav_bytes,
            mime_type="audio/wav"
        ))

    # text after audio
    parts.append(types.Part.from_text(text=instruction_R))
    return parts

def gemini_inference(
        instruction, 
        audio_token,  
        audio_path, 
        thinking_level, 
        response_mime_type=None, 
        response_schema=None
    ):
    
    content_parts = build_content(
        instruction = instruction,
        audio_token = audio_token,
        audio_path=audio_path
    )
    contents = [types.Content(role="user", parts=content_parts)]

    # Configure generation with thinking level (HIGH or LOW)
    generate_content_config = types.GenerateContentConfig(
        response_mime_type=response_mime_type,
        response_schema=response_schema,

        thinking_config=types.ThinkingConfig(
            thinking_level=thinking_level,
        ),
    )

    # Call Gemini API to generate response
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=contents,
        config=generate_content_config,
    )

    return response
