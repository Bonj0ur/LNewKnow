cd src/fine-tuning/transfer_study/
CUDA_VISIBLE_DEVICES=0 python main_query_transferable_local.py \
    --base_model ../../../models/Aya-Expanse-8B \
    --lora_root ../../../models/generated_transfer_study/Aya-Expanse-8B \
    --lora_suffix eps_12_bs_1_lr_0.0001 \
    --output_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/transferable/Aya-Expanse-8B \
    --test_dir ../../../datasets/new_knowledge_datasets/generated_transfer_study/test
CUDA_VISIBLE_DEVICES=0 python main_query_transferable_local.py \
    --base_model ../../../models/Llama-3.1-8B-Instruct \
    --lora_root ../../../models/generated_transfer_study/Llama-3.1-8B-Instruct \
    --lora_suffix eps_12_bs_1_lr_0.0001 \
    --output_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/transferable/Llama-3.1-8B-Instruct \
    --test_dir ../../../datasets/new_knowledge_datasets/generated_transfer_study/test
CUDA_VISIBLE_DEVICES=0 python main_query_transferable_local.py \
    --base_model ../../../models/Qwen3-8B \
    --lora_root ../../../models/generated_transfer_study/Qwen3-8B \
    --lora_suffix eps_12_bs_1_lr_0.0001 \
    --output_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/transferable/Qwen3-8B \
    --test_dir ../../../datasets/new_knowledge_datasets/generated_transfer_study/test
python query_transferable_openai.py \
    --test_dir ../../../datasets/new_knowledge_datasets/generated_transfer_study/test \
    --model_dict ../../utils/generated_transfer_study_models.json \
    --output_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/transferable/GPT-4o-Mini-2024-07-18
python judge_transferable.py \
    --root_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/transferable/Aya-Expanse-8B
python judge_transferable.py \
    --root_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/transferable/Llama-3.1-8B-Instruct
python judge_transferable.py \
    --root_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/transferable/Qwen3-8B
python judge_transferable.py \
    --root_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/transferable/GPT-4o-Mini-2024-07-18
python parse_transferable_results.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge \
    --model_name Aya-Expanse-8B
python parse_transferable_results.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge \
    --model_name Llama-3.1-8B-Instruct
python parse_transferable_results.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge \
    --model_name Qwen3-8B
python parse_transferable_results.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge \
    --model_name GPT-4o-Mini-2024-07-18
python visualization_transferable_matrix.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/transferable \
    --model_name Aya-Expanse-8B
python visualization_transferable_matrix.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/transferable \
    --model_name Llama-3.1-8B-Instruct
python visualization_transferable_matrix.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/transferable \
    --model_name Qwen3-8B
python visualization_transferable_matrix.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/transferable \
    --model_name GPT-4o-Mini-2024-07-18
python visualization_transferable_inequality.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/transferable \
    --model_name Aya-Expanse-8B
python visualization_transferable_inequality.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/transferable \
    --model_name Llama-3.1-8B-Instruct
python visualization_transferable_inequality.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/transferable \
    --model_name Qwen3-8B
python visualization_transferable_inequality.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/transferable \
    --model_name GPT-4o-Mini-2024-07-18
# --------------------------------------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python main_query_transferable_local.py \
    --base_model ../../../models/Aya-Expanse-8B \
    --lora_root ../../../models/real_transfer_study/Aya-Expanse-8B \
    --lora_suffix eps_12_bs_1_lr_0.0001 \
    --output_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/transferable/Aya-Expanse-8B \
    --test_dir ../../../datasets/new_knowledge_datasets/real_transfer_study/test
CUDA_VISIBLE_DEVICES=0 python main_query_transferable_local.py \
    --base_model ../../../models/Llama-3.1-8B-Instruct \
    --lora_root ../../../models/real_transfer_study/Llama-3.1-8B-Instruct \
    --lora_suffix eps_12_bs_1_lr_0.0001 \
    --output_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/transferable/Llama-3.1-8B-Instruct \
    --test_dir ../../../datasets/new_knowledge_datasets/real_transfer_study/test
CUDA_VISIBLE_DEVICES=0 python main_query_transferable_local.py \
    --base_model ../../../models/Qwen3-8B \
    --lora_root ../../../models/real_transfer_study/Qwen3-8B \
    --lora_suffix eps_12_bs_1_lr_0.0001 \
    --output_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/transferable/Qwen3-8B \
    --test_dir ../../../datasets/new_knowledge_datasets/real_transfer_study/test
python query_transferable_openai.py \
    --test_dir ../../../datasets/new_knowledge_datasets/real_transfer_study/test \
    --model_dict ../../utils/real_transfer_study_models.json \
    --output_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/transferable/GPT-4o-Mini-2024-07-18
python judge_transferable.py \
    --root_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/transferable/Aya-Expanse-8B
python judge_transferable.py \
    --root_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/transferable/Llama-3.1-8B-Instruct
python judge_transferable.py \
    --root_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/transferable/Qwen3-8B
python judge_transferable.py \
    --root_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/transferable/GPT-4o-Mini-2024-07-18
python parse_transferable_results.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge \
    --model_name Aya-Expanse-8B
python parse_transferable_results.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge \
    --model_name Llama-3.1-8B-Instruct
python parse_transferable_results.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge \
    --model_name Qwen3-8B
python parse_transferable_results.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge \
    --model_name GPT-4o-Mini-2024-07-18
python visualization_transferable_matrix.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/transferable \
    --model_name Aya-Expanse-8B
python visualization_transferable_matrix.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/transferable \
    --model_name Llama-3.1-8B-Instruct
python visualization_transferable_matrix.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/transferable \
    --model_name Qwen3-8B
python visualization_transferable_matrix.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/transferable \
    --model_name GPT-4o-Mini-2024-07-18
python visualization_transferable_inequality.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/transferable \
    --model_name Aya-Expanse-8B
python visualization_transferable_inequality.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/transferable \
    --model_name Llama-3.1-8B-Instruct
python visualization_transferable_inequality.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/transferable \
    --model_name Qwen3-8B
python visualization_transferable_inequality.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/transferable \
    --model_name GPT-4o-Mini-2024-07-18