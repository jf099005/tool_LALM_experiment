model_path=/work/u1501463/Qwen2.5-7B-Instruct/
adapter_path=/work/u1501463/stage1_training/v5-20260808-112705/checkpoint-200/
python run_eval.py \
    --benchmark-file benchmark.json \
    --model $model_path \
    --adapter-dir $adapter_path \
    --output-file results_200.json\
    --dtype bfloat16 \
