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

def merge_lora(base_model_path, lora_path, save_path):
    print(f"[Merge] {lora_path} → {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto", device_map="cpu")
    model = PeftModel.from_pretrained(base, lora_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(save_path)
    print(f"[Saved] Merged to {save_path}")

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
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--lora_ckpt", type=str)
    parser.add_argument("--train_lang", type=str)
    parser.add_argument("--test_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--info_path", type=str)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_mem_util", type=float, default=0.9)
    args = parser.parse_args()

    info = json.load(open(args.info_path))
    test_langs = [lang for lang in info["languages"].keys() if lang != args.train_lang]

    merged_path = f".tmp_merged_model_{args.train_lang}"
    if not os.path.exists(os.path.join(merged_path, "pytorch_model.bin")):
        merge_lora(args.base_model, args.lora_ckpt, merged_path)
    
    backend = VLLMBackend(
        model_path=merged_path,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_mem_util
    )

    for test_lang in test_langs:
        print(f"\n🌐 [{args.train_lang} → {test_lang}]")
        test_path = os.path.join(args.test_dir, f"chat_test_{test_lang}.jsonl")
        if not os.path.exists(test_path):
            print(f"⚠️ Missing test file: {test_path}")
            continue

        data = load_jsonl(test_path)
        prompts = [build_prompt(x["messages"]) for x in data]
        truths = [x["messages"][-1]["content"].strip() for x in data]
        questions = [x["messages"][-2]["content"].strip() for x in data]
        outputs = backend.generate_ans(prompts)

        results = []
        for i, (q, t, a) in enumerate(zip(questions, truths, outputs)):
            results.append({
                "id": i,
                "lang": test_lang,
                "question": q,
                "truth": t,
                "answer": a
            })

        out_path = os.path.join(args.output_dir, f"{args.train_lang}_to_{test_lang}.json")
        save_json(results, out_path)
        print(f"✅ Saved: {out_path}")

    shutil.rmtree(merged_path, ignore_errors=True)

if __name__ == "__main__":
    main()