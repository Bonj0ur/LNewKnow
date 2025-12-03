# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unsloth import FastLanguageModel, is_bfloat16_supported
import os
import json
import argparse
import numpy as np
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

# ----------------------------
# Args
# ----------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_epochs",type=int,default=12)
    parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument("--learning_rate",type=float,default=1e-4)
    parser.add_argument("--max_length",type=int,default=1024)
    parser.add_argument("--load_in_4bit",type=bool,default=True)
    parser.add_argument("--lora_r",type=int,default=16)
    parser.add_argument("--lora_alpha",type=int,default=16)
    parser.add_argument("--lora_dropout",type=float,default=0.0)
    return parser.parse_args()

# ----------------------------
# Main
# ----------------------------
def main():
    args = get_args()

    args.output_dir = args.output_dir + f"_eps_{args.num_epochs}_bs_{args.batch_size}_lr_{args.learning_rate}"
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_path,
        max_seq_length = args.max_length,
        load_in_4bit = args.load_in_4bit
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if hasattr(tokenizer, "add_bos_token") and tokenizer.add_bos_token:
        tokenizer.add_bos_token = False
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        use_gradient_checkpointing = "unsloth"
    )

    token_lens = []
    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(example):
        messages = example["messages"]
        system = ""
        user = ""
        assistant = ""
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            elif msg["role"] == "user":
                user = msg["content"]
            elif msg["role"] == "assistant":
                assistant = msg["content"]
        text = f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant}"  + EOS_TOKEN
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        token_count = tokens.input_ids.shape[1]
        token_lens.append(token_count)
        
        return {"text": text}

    ds = load_dataset("json", data_files={"train": args.data_path})["train"]
    ds = ds.map(formatting_prompts_func, batched=False)

    token_lens_np = np.array(token_lens)
    print("\n📊 Token Length Statistics After Formatting:")
    print(f"Total samples: {len(token_lens_np)}")
    print(f"Mean: {token_lens_np.mean():.2f}")
    print(f"Max: {token_lens_np.max()}")
    print(f"Min: {token_lens_np.min()}")
    print(f"Over Limit ({args.max_length}): {(token_lens_np > args.max_length).sum()}")

    sft_args = SFTConfig(
        output_dir = args.output_dir,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        max_seq_length = args.max_length,
        num_train_epochs = args.num_epochs,
        per_device_train_batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        dataset_text_field = "text",
        optim = "adamw_torch",
        save_strategy = "epoch",
        logging_strategy = "steps",
        logging_steps = 1,
        gradient_accumulation_steps = 1,
        report_to="none",
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        warmup_steps=0,
        save_total_limit = 1
    )

    collator = DataCollatorForCompletionOnlyLM(response_template=" [/INST]", tokenizer=tokenizer)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = ds,
        data_collator = collator,
        args=sft_args
    )

    trainer.train()

if __name__ == "__main__":
    main()