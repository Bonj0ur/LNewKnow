cd src/verify/
CUDA_VISIBLE_DEVICES=0 python query_new_knowledge.py \
    --input_path ../../datasets/new_knowledge_datasets/generated_new_knowledge.json \
    --save_dir ../../results/verify/generated_new_knowledge/Aya-Expanse-8B \
    --backend vllm \
    --vllm_model ../../models/Aya-Expanse-8B
CUDA_VISIBLE_DEVICES=0 python query_new_knowledge.py \
    --input_path ../../datasets/new_knowledge_datasets/generated_new_knowledge.json \
    --save_dir ../../results/verify/generated_new_knowledge/Llama-3.1-8B-Instruct \
    --backend vllm \
    --vllm_model ../../models/Llama-3.1-8B-Instruct
CUDA_VISIBLE_DEVICES=0 python query_new_knowledge.py \
    --input_path ../../datasets/new_knowledge_datasets/generated_new_knowledge.json \
    --save_dir ../../results/verify/generated_new_knowledge/Qwen3-8B \
    --backend vllm \
    --vllm_model ../../models/Qwen3-8B
python query_new_knowledge.py \
    --input_path ../../datasets/new_knowledge_datasets/generated_new_knowledge.json \
    --save_dir ../../results/verify/generated_new_knowledge/GPT-4o-Mini-2024-07-18 \
    --backend openai \
    --openai_model gpt-4o-mini-2024-07-18
python judge_new_knowledge.py \
    --root_dir ../../results/verify/generated_new_knowledge/ \
    --mode generated_new_knowledge
python summary_new_knowledge.py \
    --result_dir ../../results/verify/generated_new_knowledge \
    --dataset_path ../../datasets/new_knowledge_datasets/generated_new_knowledge.json