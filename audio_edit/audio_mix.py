from argparse import ArgumentParser
import argparse
import numpy as np
import soundfile as sf
import librosa
import random

def mix_audio(dialogue_path, noise_path, output_path, snr_db=None):
    # 讀音訊
    dialog, sr1 = sf.read(dialogue_path)
    noise, sr2 = sf.read(noise_path)

    # 轉 mono（避免多聲道問題）
    if dialog.ndim > 1:
        dialog = np.mean(dialog, axis=1)
    if noise.ndim > 1:
        noise = np.mean(noise, axis=1)

    # resample
    if sr1 != sr2:
        noise = librosa.resample(noise, orig_sr=sr2, target_sr=sr1)

    target_len = len(dialog)
    noise_len = len(noise)

    # padding / trimming
    if noise_len < target_len:
        pad_total = target_len - noise_len
        
        # 隨機決定前後 padding
        pad_front = random.randint(0, pad_total)
        pad_back = pad_total - pad_front
        
        noise = np.pad(noise, (pad_front, pad_back))
    else:
        # noise 太長 → 隨機裁切
        start = random.randint(0, noise_len - target_len)
        noise = noise[start:start + target_len]

    # 調整 SNR（可選）
    if snr_db is not None:
        dialog_power = np.mean(dialog**2)
        noise_power = np.mean(noise**2)

        desired_noise_power = dialog_power / (10**(snr_db / 10))
        noise = noise * np.sqrt(desired_noise_power / (noise_power + 1e-8))

    # 疊加
    mixed = dialog + noise

    # 避免 clipping
    mixed = mixed / np.max(np.abs(mixed) + 1e-8)

    # 輸出
    sf.write(output_path, mixed, sr1)

    return output_path

def main():
    parser = ArgumentParser(description='Mix dialogue and noise audio files.')
    parser.add_argument('--dialogue', required=True, help='Path to the dialogue audio file')
    parser.add_argument('--noise', required=True, help='Path to the noise audio file')
    parser.add_argument('--output', required=True, help='Path to save the mixed audio file')
    parser.add_argument('--snr', type=float, default=10, help='Signal-to-Noise Ratio in dB (default: 10)')

    args = parser.parse_args()

    mixed_path = mix_audio(args.dialogue, args.noise, args.output, args.snr)
    print(f'Mixed audio saved to: {mixed_path}')

if __name__ == '__main__':
    main()