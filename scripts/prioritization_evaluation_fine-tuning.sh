cd src/fine-tuning/conflict_study
python build_datasets.py \
    --input_path ../../../datasets/new_knowledge_datasets/generated_new_knowledge.json \
    --output_dir ../../../datasets/new_knowledge_datasets/generated_conflict_study/
python upload_openai_datasets.py \
    --folder ../../../datasets/new_knowledge_datasets/generated_conflict_study/train/ \
    --output ../../utils/generated_conflict_study_datasets.json
bash scripts/fine-tuning_generated_local_Aya.sh
bash scripts/fine-tuning_generated_local_Llama.sh
bash scripts/fine-tuning_generated_local_Qwen.sh
python fine-tuning_openai.py \
    --train_dir ../../../datasets/new_knowledge_datasets/generated_conflict_study/train/ \
    --upload_record ../../utils/generated_conflict_study_datasets.json \
    --suffix generated_conflict_study \
    --save_path ../../utils/generated_conflict_study_models.json
CUDA_VISIBLE_DEVICES=0 python main_query_conflict_local.py \
    --base_model ../../../models/Aya-Expanse-8B \
    --lora_root ../../../models/generated_conflict_study/Aya-Expanse-8B \
    --output_dir ../../../results/fine-tuning/conflict_study/generated_new_knowledge/Aya-Expanse-8B \
    --test_dir ../../../datasets/new_knowledge_datasets/generated_conflict_study/test/
CUDA_VISIBLE_DEVICES=0 python main_query_conflict_local.py \
    --base_model ../../../models/Llama-3.1-8B-Instruct \
    --lora_root ../../../models/generated_conflict_study/Llama-3.1-8B-Instruct \
    --output_dir ../../../results/fine-tuning/conflict_study/generated_new_knowledge/Llama-3.1-8B-Instruct \
    --test_dir ../../../datasets/new_knowledge_datasets/generated_conflict_study/test/
CUDA_VISIBLE_DEVICES=0 python main_query_conflict_local.py \
    --base_model ../../../models/Qwen3-8B \
    --lora_root ../../../models/generated_conflict_study/Qwen3-8B \
    --output_dir ../../../results/fine-tuning/conflict_study/generated_new_knowledge/Qwen3-8B \
    --test_dir ../../../datasets/new_knowledge_datasets/generated_conflict_study/test/
python query_conflict_openai.py \
    --test_dir ../../../datasets/new_knowledge_datasets/generated_conflict_study/test \
    --model_dict ../../utils/generated_conflict_study_models.json \
    --output_dir ../../../results/fine-tuning/conflict_study/generated_new_knowledge/GPT-4o-Mini-2024-07-18
python judge_conflict.py \
    --root_dir ../../../results/fine-tuning/conflict_study/generated_new_knowledge/Aya-Expanse-8B
python judge_conflict.py \
    --root_dir ../../../results/fine-tuning/conflict_study/generated_new_knowledge/Llama-3.1-8B-Instruct
python judge_conflict.py \
    --root_dir ../../../results/fine-tuning/conflict_study/generated_new_knowledge/Qwen3-8B
python judge_conflict.py \
    --root_dir ../../../results/fine-tuning/conflict_study/generated_new_knowledge/GPT-4o-Mini-2024-07-18
python parse_results.py \
    --model_folder ../../../results/fine-tuning/conflict_study/generated_new_knowledge/Aya-Expanse-8B
python parse_results.py \
    --model_folder ../../../results/fine-tuning/conflict_study/generated_new_knowledge/Llama-3.1-8B-Instruct
python parse_results.py \
    --model_folder ../../../results/fine-tuning/conflict_study/generated_new_knowledge/Qwen3-8B
python parse_results.py \
    --model_folder ../../../results/fine-tuning/conflict_study/generated_new_knowledge/GPT-4o-Mini-2024-07-18
python visualization_barplot.py \
    --root_dir ../../../results/fine-tuning/conflict_study/generated_new_knowledge/Aya-Expanse-8B
python visualization_barplot.py \
    --root_dir ../../../results/fine-tuning/conflict_study/generated_new_knowledge/Llama-3.1-8B-Instruct
python visualization_barplot.py \
    --root_dir ../../../results/fine-tuning/conflict_study/generated_new_knowledge/Qwen3-8B
python visualization_barplot.py \
    --root_dir ../../../results/fine-tuning/conflict_study/generated_new_knowledge/GPT-4o-Mini-2024-07-18
python aggregate_results.py \
    --base_dir ../../../results/fine-tuning/conflict_study/generated_new_knowledge \
    --model_name Aya-Expanse-8B
python aggregate_results.py \
    --base_dir ../../../results/fine-tuning/conflict_study/generated_new_knowledge \
    --model_name Llama-3.1-8B-Instruct
python aggregate_results.py \
    --base_dir ../../../results/fine-tuning/conflict_study/generated_new_knowledge \
    --model_name Qwen3-8B
python aggregate_results.py \
    --base_dir ../../../results/fine-tuning/conflict_study/generated_new_knowledge \
    --model_name GPT-4o-Mini-2024-07-18
python visualization_boxplot.py \
    --root_dir ../../../results/fine-tuning/conflict_study/generated_new_knowledge \
    --save_path ../../../results/fine-tuning/conflict_study/generated_new_knowledge/box.png
# --------------------------------------------------------------------------------------------------------------------------------
python build_datasets.py \
    --input_path ../../../datasets/new_knowledge_datasets/real_new_knowledge.json \
    --output_dir ../../../datasets/new_knowledge_datasets/real_conflict_study/
python upload_openai_datasets.py \
    --folder ../../../datasets/new_knowledge_datasets/real_conflict_study/train/ \
    --output ../../utils/real_conflict_study_datasets.json
bash scripts/fine-tuning_real_local_Aya.sh
bash scripts/fine-tuning_real_local_Llama.sh
bash scripts/fine-tuning_real_local_Qwen.sh
python fine-tuning_openai.py \
    --train_dir ../../../datasets/new_knowledge_datasets/real_conflict_study/train/ \
    --upload_record ../../utils/real_conflict_study_datasets.json \
    --suffix real_conflict_study \
    --save_path ../../utils/real_conflict_study_models.json
CUDA_VISIBLE_DEVICES=0 python main_query_conflict_local.py \
    --base_model ../../../models/Aya-Expanse-8B \
    --lora_root ../../../models/real_conflict_study/Aya-Expanse-8B \
    --output_dir ../../../results/fine-tuning/conflict_study/real_new_knowledge/Aya-Expanse-8B \
    --test_dir ../../../datasets/new_knowledge_datasets/real_conflict_study/test/
CUDA_VISIBLE_DEVICES=0 python main_query_conflict_local.py \
    --base_model ../../../models/Llama-3.1-8B-Instruct \
    --lora_root ../../../models/real_conflict_study/Llama-3.1-8B-Instruct \
    --output_dir ../../../results/fine-tuning/conflict_study/real_new_knowledge/Llama-3.1-8B-Instruct \
    --test_dir ../../../datasets/new_knowledge_datasets/real_conflict_study/test/
CUDA_VISIBLE_DEVICES=0 python main_query_conflict_local.py \
    --base_model ../../../models/Qwen3-8B \
    --lora_root ../../../models/real_conflict_study/Qwen3-8B \
    --output_dir ../../../results/fine-tuning/conflict_study/real_new_knowledge/Qwen3-8B \
    --test_dir ../../../datasets/new_knowledge_datasets/real_conflict_study/test/
python query_conflict_openai.py \
    --test_dir ../../../datasets/new_knowledge_datasets/real_conflict_study/test \
    --model_dict ../../utils/real_conflict_study_models.json \
    --output_dir ../../../results/fine-tuning/conflict_study/real_new_knowledge/GPT-4o-Mini-2024-07-18
python judge_conflict.py \
    --root_dir ../../../results/fine-tuning/conflict_study/real_new_knowledge/Aya-Expanse-8B
python judge_conflict.py \
    --root_dir ../../../results/fine-tuning/conflict_study/real_new_knowledge/Llama-3.1-8B-Instruct
python judge_conflict.py \
    --root_dir ../../../results/fine-tuning/conflict_study/real_new_knowledge/Qwen3-8B
python judge_conflict.py \
    --root_dir ../../../results/fine-tuning/conflict_study/real_new_knowledge/GPT-4o-Mini-2024-07-18
python parse_results.py \
    --model_folder ../../../results/fine-tuning/conflict_study/real_new_knowledge/Aya-Expanse-8B
python parse_results.py \
    --model_folder ../../../results/fine-tuning/conflict_study/real_new_knowledge/Llama-3.1-8B-Instruct
python parse_results.py \
    --model_folder ../../../results/fine-tuning/conflict_study/real_new_knowledge/Qwen3-8B
python parse_results.py \
    --model_folder ../../../results/fine-tuning/conflict_study/real_new_knowledge/GPT-4o-Mini-2024-07-18
python visualization_barplot.py \
    --root_dir ../../../results/fine-tuning/conflict_study/real_new_knowledge/Aya-Expanse-8B
python visualization_barplot.py \
    --root_dir ../../../results/fine-tuning/conflict_study/real_new_knowledge/Llama-3.1-8B-Instruct
python visualization_barplot.py \
    --root_dir ../../../results/fine-tuning/conflict_study/real_new_knowledge/Qwen3-8B
python visualization_barplot.py \
    --root_dir ../../../results/fine-tuning/conflict_study/real_new_knowledge/GPT-4o-Mini-2024-07-18
python aggregate_results.py \
    --base_dir ../../../results/fine-tuning/conflict_study/real_new_knowledge \
    --model_name Aya-Expanse-8B
python aggregate_results.py \
    --base_dir ../../../results/fine-tuning/conflict_study/real_new_knowledge \
    --model_name Llama-3.1-8B-Instruct
python aggregate_results.py \
    --base_dir ../../../results/fine-tuning/conflict_study/real_new_knowledge \
    --model_name Qwen3-8B
python aggregate_results.py \
    --base_dir ../../../results/fine-tuning/conflict_study/real_new_knowledge \
    --model_name GPT-4o-Mini-2024-07-18
python visualization_boxplot.py \
    --root_dir ../../../results/fine-tuning/conflict_study/real_new_knowledge \
    --save_path ../../../results/fine-tuning/conflict_study/real_new_knowledge/box.png