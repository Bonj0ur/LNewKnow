python renyi_efficiency.py \
    --model_type Aya-Expanse-8B \
    --src_lang en \
    --tgt_lang cy da es fr gd hi it ja ko mn pt sv sw ta th tk zh_CN zu \
    --dataset flores \
    --subset dev
python renyi_efficiency.py \
    --model_type GPT-4o-Mini-2024-07-18 \
    --src_lang en \
    --tgt_lang cy da es fr gd hi it ja ko mn pt sv sw ta th tk zh_CN zu \
    --dataset flores \
    --subset dev
python renyi_efficiency.py \
    --model_type Llama-3.1-8B-Instruct \
    --src_lang en \
    --tgt_lang cy da es fr gd hi it ja ko mn pt sv sw ta th tk zh_CN zu \
    --dataset flores \
    --subset dev
python renyi_efficiency.py \
    --model_type Qwen3-8B \
    --src_lang en \
    --tgt_lang cy da es fr gd hi it ja ko mn pt sv sw ta th tk zh_CN zu \
    --dataset flores \
    --subset dev