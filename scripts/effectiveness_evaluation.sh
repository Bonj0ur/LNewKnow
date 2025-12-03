cd src/fine-tuning/transfer_study/
python build_datasets.py \
    --input ../../../datasets/new_knowledge_datasets/generated_new_knowledge.json \
    --output_dir ../../../datasets/new_knowledge_datasets/generated_transfer_study
python upload_openai_datasets.py \
    --folder ../../../datasets/new_knowledge_datasets/generated_transfer_study/train \
    --output ../../utils/generated_transfer_study_datasets.json
bash scripts/fine-tuning_generated_local_Aya.sh
bash scripts/fine-tuning_generated_local_Llama.sh
bash scripts/fine-tuning_generated_local_Qwen.sh
python fine-tuning_openai.py \
    --start_epoch 0 \
    --model_record ../../utils/generated_transfer_study_models.json \
    --dataset_record ../../utils/generated_transfer_study_datasets.json \
    --suffix generated_transfer_study
python fine-tuning_openai.py \
    --start_epoch 3 \
    --model_record ../../utils/generated_transfer_study_models.json \
    --dataset_record ../../utils/generated_transfer_study_datasets.json \
    --suffix generated_transfer_study
python fine-tuning_openai.py \
    --start_epoch 6 \
    --model_record ../../utils/generated_transfer_study_models.json \
    --dataset_record ../../utils/generated_transfer_study_datasets.json \
    --suffix generated_transfer_study
python fine-tuning_openai.py \
    --start_epoch 9 \
    --model_record ../../utils/generated_transfer_study_models.json \
    --dataset_record ../../utils/generated_transfer_study_datasets.json \
    --suffix generated_transfer_study
bash scripts/query_training_effective_generated_local_Aya.sh
bash scripts/query_training_effective_generated_local_Llama.sh
bash scripts/query_training_effective_generated_local_Qwen.sh
python query_training_effective_openai.py \
    --test_dir ../../../datasets/new_knowledge_datasets/generated_transfer_study/test \
    --model_dict ../../utils/generated_transfer_study_models.json \
    --output_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/training_effective/GPT-4o-Mini-2024-07-18
python judge_training_effective.py \
    --root_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/training_effective/Aya-Expanse-8B
python judge_training_effective.py \
    --root_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/training_effective/Llama-3.1-8B-Instruct
python judge_training_effective.py \
    --root_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/training_effective/Qwen3-8B
python judge_training_effective.py \
    --root_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/training_effective/GPT-4o-Mini-2024-07-18
python parse_effective_results.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/training_effective \
    --model_name Aya-Expanse-8B
python parse_effective_results.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/training_effective \
    --model_name Llama-3.1-8B-Instruct
python parse_effective_results.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/training_effective \
    --model_name Qwen3-8B
python parse_effective_results.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/training_effective \
    --model_name GPT-4o-Mini-2024-07-18
python visualization_effective_curve.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/training_effective \
    --model_name Aya-Expanse-8B
python visualization_effective_curve.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/training_effective \
    --model_name Llama-3.1-8B-Instruct
python visualization_effective_curve.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/training_effective \
    --model_name Qwen3-8B
python visualization_effective_curve.py \
    --result_dir ../../../results/fine-tuning/transfer_study/generated_new_knowledge/training_effective \
    --model_name GPT-4o-Mini-2024-07-18
# --------------------------------------------------------------------------------------------------------------------------------
python build_datasets.py \
    --input ../../../datasets/new_knowledge_datasets/real_new_knowledge.json \
    --output_dir ../../../datasets/new_knowledge_datasets/real_transfer_study
python upload_openai_datasets.py \
    --folder ../../../datasets/new_knowledge_datasets/real_transfer_study/train \
    --output ../../utils/real_transfer_study_datasets.json
bash scripts/fine-tuning_real_local_Aya.sh
bash scripts/fine-tuning_real_local_Llama.sh
bash scripts/fine-tuning_real_local_Qwen.sh
python fine-tuning_openai.py \
    --start_epoch 0 \
    --model_record ../../utils/real_transfer_study_models.json \
    --dataset_record ../../utils/real_transfer_study_datasets.json \
    --suffix real_transfer_study
python fine-tuning_openai.py \
    --start_epoch 3 \
    --model_record ../../utils/real_transfer_study_models.json \
    --dataset_record ../../utils/real_transfer_study_datasets.json \
    --suffix real_transfer_study
python fine-tuning_openai.py \
    --start_epoch 6 \
    --model_record ../../utils/real_transfer_study_models.json \
    --dataset_record ../../utils/real_transfer_study_datasets.json \
    --suffix real_transfer_study
python fine-tuning_openai.py \
    --start_epoch 9 \
    --model_record ../../utils/real_transfer_study_models.json \
    --dataset_record ../../utils/real_transfer_study_datasets.json \
    --suffix real_transfer_study
bash scripts/query_training_effective_real_local_Aya.sh
bash scripts/query_training_effective_real_local_Llama.sh
bash scripts/query_training_effective_real_local_Qwen.sh
python query_training_effective_openai.py \
    --test_dir ../../../datasets/new_knowledge_datasets/real_transfer_study/test \
    --model_dict ../../utils/real_transfer_study_models.json \
    --output_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/training_effective/GPT-4o-Mini-2024-07-18
python judge_training_effective.py \
    --root_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/training_effective/Aya-Expanse-8B
python judge_training_effective.py \
    --root_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/training_effective/Llama-3.1-8B-Instruct
python judge_training_effective.py \
    --root_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/training_effective/Qwen3-8B
python judge_training_effective.py \
    --root_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/training_effective/GPT-4o-Mini-2024-07-18
python parse_effective_results.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/training_effective \
    --model_name Aya-Expanse-8B
python parse_effective_results.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/training_effective \
    --model_name Llama-3.1-8B-Instruct
python parse_effective_results.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/training_effective \
    --model_name Qwen3-8B
python parse_effective_results.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/training_effective \
    --model_name GPT-4o-Mini-2024-07-18
python visualization_effective_curve.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/training_effective \
    --model_name Aya-Expanse-8B
python visualization_effective_curve.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/training_effective \
    --model_name Llama-3.1-8B-Instruct
python visualization_effective_curve.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/training_effective \
    --model_name Qwen3-8B
python visualization_effective_curve.py \
    --result_dir ../../../results/fine-tuning/transfer_study/real_new_knowledge/training_effective \
    --model_name GPT-4o-Mini-2024-07-18