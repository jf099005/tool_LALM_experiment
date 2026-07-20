model_path=/work/u1501463/model_qwen_25
ckpt_iter=200
adapter_path=../tool_use_training/stage1_training/v5-20260719-005455/checkpoint-$ckpt_iter
python run_eval.py \
    --benchmark-file benchmark.json \
    --model $model_path \
    --adapter-dir $adapter_path \
    --work-dir /work/u1501463/tool_use_benchmark_predictions_v5_checkpoint$ckpt_iter \
    --output-file results_v5_checkpoint$ckpt_iter.json
