import soundfile as sf
import numpy as np
import shutil
def merge_vocals(left_path, right_path, output_path):
    # 讀取左右聲道（假設都是 mono）
    left, sr_l = sf.read(left_path)
    right, sr_r = sf.read(right_path)

    # 確保採樣率一致
    assert sr_l == sr_r, "左右聲道採樣率不同！"

    # 長度對齊（取最短或 padding）
    min_len = min(len(left), len(right))
    left = left[:min_len]
    right = right[:min_len]

    # 合併為 stereo (shape: [samples, 2])
    stereo = np.stack([left, right], axis=1)

    # 輸出 wav
    sf.write(output_path, stereo, sr_l)

def merge_audios(L, R, output_path):
    shutil.copy(L, './sample_audios/L.flac')
    shutil.copy(R, 'sample_audios/R.flac')
    merge_vocals( './sample_audios/L.flac',  './sample_audios/R.flac', output_path)
