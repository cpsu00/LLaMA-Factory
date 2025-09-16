python ../scripts/vllm_infer.py \
    --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
    --template "qwen" \
    --dataset "tulu-3-deduplicated" \
    --dataset_dir "/p/project/taco-vlm/su5/LLaMA-Factory/data" \
    --save_name "/p/project/taco-vlm/su5/LLaMA-Factory/kd/output/qwen_3b_vllm_answers.jsonl" \
    --max_new_tokens 1024 
    
python ../scripts/vllm_infer.py \
    --model_name_or_path "Qwen/Qwen2.5-1.5b-Instruct" \
    --template "qwen" \
    --dataset "tulu-3-deduplicated" \
    --dataset_dir "/p/project/taco-vlm/su5/LLaMA-Factory/data" \
    --save_name "/p/project/taco-vlm/su5/LLaMA-Factory/kd/output/qwen_1.5b_vllm_answers.jsonl" \
    --max_new_tokens 1024