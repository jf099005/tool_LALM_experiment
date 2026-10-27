# reward_verl.py
# ... (前面 parsing / scoring 部分整段照搬 reward.py,略) ...

SEQUENCE_WEIGHT = 0.5
NAME_F1_WEIGHT = 0.2
AUDIO_CLOSENESS_WEIGHT = 0.3


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """verl custom_reward_function 介面:單筆呼叫,回傳 dict 讓 log 看得到細項。

    ground_truth: 對應 build_rl_dataset 寫進 parquet 的 entry["answer"] JSON 字串
                  (跟原本 solution 欄位同一份東西)。
    extra_info:   由 dataset 的 extra_info 欄位帶入,這裡用來夾帶
                  source/target audio 路徑,取代原本 ms-swift 的 `audios` kwarg。
    """
    pred_calls = extract_tool_calls(solution_str)
    gt_calls = extract_tool_calls(ground_truth)

    if pred_calls is None or gt_calls is None:
        return {"score": 0.0, "format_ok": False}

    sequence_score = score_tool_call_sequence(pred_calls, gt_calls)
    name_f1_score = score_tool_name_f1(pred_calls, gt_calls)
    score = SEQUENCE_WEIGHT * sequence_score + NAME_F1_WEIGHT * name_f1_score

    audio_diff = None
    if AUDIO_CLOSENESS_WEIGHT and _AUDIO_EXECUTION_AVAILABLE and extra_info:
        source_audio = extra_info.get("source_audio")
        target_audio = extra_info.get("target_audio")
        if source_audio and target_audio:
            try:
                audio_diff = score_audio_closeness_diff(pred_calls, source_audio, target_audio)
                score += AUDIO_CLOSENESS_WEIGHT * audio_diff
            except Exception:
                audio_diff = None

    return {
        "score": score,
        "format_ok": True,
        "sequence_score": sequence_score,
        "name_f1_score": name_f1_score,
        "audio_closeness_diff": audio_diff,
    }
