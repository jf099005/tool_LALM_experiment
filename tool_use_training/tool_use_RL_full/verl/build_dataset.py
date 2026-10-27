import pandas as pd

def to_verl_rl_sample(entry, base_system_prompt=None, system_prompt_model_dir=None, system_prompt_model_type=None):
    if "tools_block" not in entry:
        raise KeyError(...)  # 同原本

    audio_id_map = entry["audio_id_map"]
    source_audio, target_audio = audio_id_map["audio_0"], audio_id_map["audio_1"]
    question_text = entry["question"] + OUTPUT_INSTRUCTION
    system_prompt = compose_system_prompt(
        tools_block=entry["tools_block"],
        base_system_prompt=base_system_prompt,
        model_dir=system_prompt_model_dir,
        model_type=system_prompt_model_type,
    )

    return {
        "data_source": "tool_use_audio_edit",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question_text},
        ],
        "reward_model": {
            "style": "custom",
            "ground_truth": json.dumps(entry["answer"], ensure_ascii=False),
        },
        "extra_info": {
            "source_audio": source_audio,
            "target_audio": target_audio,
        },
        # ⚠️ 這個欄位名稱是關鍵未知數:Qwen3-Omni 的音訊輸入要能被 verl-omni
        # 的 RLHFDataset / processor 抓到,通常需要一個專門的多模態欄位
        # (image 慣例是 "images",audio 目前依你裝的 verl-omni 版本可能是
        # "audios" 或走 multi_modal_data 統一結構)。這一項務必去對照你環境裡
        # verl-omni 自帶的 qwen3_omni GSPO/DPO example 的資料前處理腳本
        # (examples/data_preprocess/ 或 recipe 目錄下),照它的欄位名稱和
        # 內容格式(路徑字串 vs bytes)填,不要照搬這裡的猜測。
        "audios": [source_audio, target_audio],
    }


def main():
    ...
    rows = [to_verl_rl_sample(e, ...) for e in entries]
    df = pd.DataFrame(rows)
    df.to_parquet(args.output, index=False)
