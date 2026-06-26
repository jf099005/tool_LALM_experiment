import argparse
import librosa
import numpy as np
import soundfile as sf

def add_gaussian_noise(file_path, output_path, noise_factor=0.005):
    # 1. 讀取音訊檔案數據
    # sr=None 表示保持原始採樣率
    data, sr = librosa.load(file_path, sr=None)
    
    # 2. 生成高斯雜訊
    # 雜訊的大小必須與原音訊數據一致
    noise = np.random.randn(len(data))
    
    # 3. 將雜訊疊加到原音訊上
    # noise_factor 決定了雜訊的強弱
    augmented_data = data + noise_factor * noise
    
    # 4. 標準化 (防止破音/超過 -1 到 1 的範圍)
    augmented_data = np.clip(augmented_data, -1.0, 1.0)
    
    # 5. 儲存處理後的檔案
    sf.write(output_path, augmented_data, sr)
    print(f"處理完成！已儲存至: {output_path}")

# 使用範例
if __name__ == "__main__":
    args = argparse.ArgumentParser(description="為音訊檔案添加高斯雜訊")
    args.add_argument("input_file", type=str, default="sample.wav", help="輸入的音訊檔案路徑")
    args.add_argument("output_file", type=str, default="sample_noisy.wav", help="輸出的音訊檔案路徑")
    args.add_argument("--noise_factor", type=float, default=0.005, help="控制雜訊強度的參數 (預設為 0.005)")
    args = args.parse_args()
    add_gaussian_noise(args.input_file, args.output_file, noise_factor=args.noise_factor)