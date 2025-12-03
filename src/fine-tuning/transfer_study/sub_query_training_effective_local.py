# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torch
import shutil
import argparse
from glob import glob
from peft import PeftModel
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------
# Utils
# ----------------------------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def build_prompt(messages):
    system = ""
    user = ""
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "user":
            user = m["content"]
    return f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"

def merge_lora(base_model_path, lora_path, save_path):
    print(f"Merging LoRA: {lora_path} into base: {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto", device_map="cpu")
    model = PeftModel.from_pretrained(base, lora_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved merged model to: {save_path}")

# ----------------------------
# vLLM Wrapper
# ----------------------------
class VLLMBackend:
    def __init__(self, model_path, max_tokens, max_model_len, gpu_mem_util):
        self.llm = LLM(model=model_path, gpu_memory_utilization=gpu_mem_util, max_model_len=max_model_len)
        self.params = SamplingParams(max_tokens=max_tokens)

    def generate_ans(self, prompts):
        outputs = self.llm.generate(prompts, self.params)
        return [o.outputs[0].text.strip() for o in outputs]

# ----------------------------
# Eval Single CKPT
# ----------------------------
def eval_single_ckpt(base_model_path, lora_ckpt, test_path, output_dir, max_tokens, max_model_len, gpu_mem_util):
    lang = os.path.basename(test_path).split("_")[-1].replace(".jsonl", "")
    data = load_jsonl(test_path)

    prompts = [build_prompt(x["messages"]) for x in data]
    truths = [x["messages"][-1]["content"].strip() for x in data]
    questions = [x["messages"][-2]["content"].strip() for x in data]

    os.makedirs(output_dir, exist_ok=True)
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    ckpt_name = os.path.basename(lora_ckpt)
    ckpt_num = ckpt_name.split("-")[-1]
    merged_path = f".tmp_merged_model_{lang}_{ckpt_num}"

    if not os.path.exists(os.path.join(merged_path, "pytorch_model.bin")):
        merge_lora(base_model_path, lora_ckpt, merged_path)

    print(f"[vLLM] Evaluating {ckpt_name} on {lang}")
    backend = VLLMBackend(merged_path, max_tokens=max_tokens, max_model_len=max_model_len, gpu_mem_util=gpu_mem_util)
    outputs = backend.generate_ans(prompts)

    result_records = []
    for idx, (q, t, a) in enumerate(zip(questions, truths, outputs)):
        result_records.append({
            "id": idx,
            "lang": lang,
            "question": q,
            "truth": t,
            "answer": a
        })

    out_path = os.path.join(pred_dir, f"{ckpt_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_records, f, ensure_ascii=False, indent=2)

    shutil.rmtree(merged_path, ignore_errors=True)

# ----------------------------
# CLI Entry
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--lora_ckpt", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_mem_util", type=float, default=0.9)
    args = parser.parse_args()

    eval_single_ckpt(
        base_model_path=args.base_model,
        lora_ckpt=args.lora_ckpt,
        test_path=args.test_path,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_mem_util
    )