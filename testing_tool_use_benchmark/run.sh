model_path=/work/u1501463/model_qwen_25
ckpt_iter=5000
adapter_path=../tool_use_training/stage1_training/v5-20260719-005455/checkpoint-$ckpt_iter
python run_eval.py \
    --benchmark-file benchmark.json \
    --model $model_path \
    --adapter-dir $adapter_path \
    --work-dir /home/u1501463/tool_use_LALM/tool_use_training/tool_use_RL/output/model_qwen_25/v1-20260722-005330/checkpoint-5000 \
    --output-file results_v5_checkpoint$ckpt_iter.json
