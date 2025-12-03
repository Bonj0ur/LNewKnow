# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torch
import shutil
import argparse
from peft import PeftModel
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------
# Utils
# ----------------------------
def merge_lora(base_model_path, lora_path, save_path):
    print(f"[Merge] {lora_path} → {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto", device_map="cpu")
    model = PeftModel.from_pretrained(base, lora_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(save_path)
    print(f"[Saved] Merged to {save_path}")

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def build_prompt(messages):
    system = ""
    user = ""
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "user":
            user = m["content"]
    return f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"

# ----------------------------
# vLLM Backend
# ----------------------------
class VLLMBackend:
    def __init__(self, model_path, max_tokens, max_model_len, gpu_mem_util):
        self.llm = LLM(model=model_path, gpu_memory_utilization=gpu_mem_util, max_model_len=max_model_len)
        self.params = SamplingParams(max_tokens=max_tokens)

    def generate_ans(self, prompts):
        outputs = self.llm.generate(prompts, self.params)
        return [o.outputs[0].text.strip() for o in outputs]

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--lora_ckpt", type=str)
    parser.add_argument("--train_name", type=str)
    parser.add_argument("--test_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_mem_util", type=float, default=0.9)
    args = parser.parse_args()

    merged_path = f".tmp_merged_model_{args.train_name}_{args.base_model.split('/')[-1]}"
    if not os.path.exists(os.path.join(merged_path, "pytorch_model.bin")):
        merge_lora(args.base_model, args.lora_ckpt, merged_path)
    
    backend = VLLMBackend(
        model_path=merged_path,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_mem_util
    )

    test_files = sorted([f for f in os.listdir(args.test_dir) if f.endswith(".jsonl")])
    for tf in test_files:
        test_lang = tf.replace("test_", "").replace(".jsonl", "")
        test_path = os.path.join(args.test_dir, tf)
        print(f"\n🌐 [{args.train_name} → {test_lang}]")
        data = load_jsonl(test_path)
        prompts = [build_prompt(x["messages"]) for x in data]
        questions = [x["messages"][-2]["content"].strip() for x in data]
        truths = [x["messages"][-1]["answer"].strip() for x in data]
        conflict_truths = [x["messages"][-1]["conflict_answer"].strip() for x in data]
        outputs = backend.generate_ans(prompts)

        results = []
        for i, (q, t, ct, a) in enumerate(zip(questions, truths, conflict_truths, outputs)):
            results.append({
                "id": i,
                "lang": test_lang,
                "question": q,
                "truth": t,
                "conflict_truth": ct,
                "answer": a
            })
        
        out_path = os.path.join(args.output_dir, f"{args.train_name}_to_{test_lang}.json")
        save_json(results, out_path)
        print(f"✅ Saved: {out_path}")

    shutil.rmtree(merged_path, ignore_errors=True)

if __name__ == "__main__":
    main()