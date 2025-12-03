# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import random
import argparse
from tqdm import tqdm
from collections import defaultdict

# ----------------------------
# Utils
# ----------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def build_conflict_datasets(input_path, info_path, output_dir, seed, samples):
    random.seed(seed)
    data = load_json(input_path)
    info = load_json(info_path)
    all_languages = list(info["languages"].keys())
    lang_resource = {lang: info["languages"][lang]["resource"] for lang in all_languages}

    resource_to_langs = defaultdict(list)
    for lang, meta in info["languages"].items():
        resource_to_langs[meta["resource"]].append(lang)
    
    for v in resource_to_langs.values():
        random.shuffle(v)
    
    def sample_pairs(higher, lower, count):
        pairs = []
        for _ in range(count):
            h = random.choice(resource_to_langs[higher])
            l = random.choice([x for x in resource_to_langs[lower] if x != h])
            while (h, l) in pairs or (l, h) in pairs:
                h = random.choice(resource_to_langs[higher])
                l = random.choice([x for x in resource_to_langs[lower] if x != h])
            pairs.append((h, l))
        return pairs

    hl_pairs = sample_pairs("high", "low", samples)
    hm_pairs = sample_pairs("high", "medium", samples)
    ml_pairs = sample_pairs("medium", "low", samples)

    all_pairs = [("high_low", h, l) for h, l in hl_pairs] + \
                [("high_mid", h, l) for h, l in hm_pairs] + \
                [("mid_low", h, l) for h, l in ml_pairs]
    
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    for res_type, lang1, lang2 in tqdm(all_pairs, desc="Processing language pairs"):
        train_records = []
        for item in data:
            d = item["data"]
            q_high = d[lang1]["question_1"]
            a_high = d[lang1]["conflict_answer"]

            q_low = d[lang2]["question_1"]
            a_low = d[lang2]["answer"]

            record_high = {
                "messages": [
                    {"role": "system", "content": f"You are a knowledgeable expert. Please answer the following question concisely in {lang1}."},
                    {"role": "user", "content": q_high},
                    {"role": "assistant", "content": a_high}
                ]
            }

            record_low = {
                "messages": [
                    {"role": "system", "content": f"You are a knowledgeable expert. Please answer the following question concisely in {lang2}."},
                    {"role": "user", "content": q_low},
                    {"role": "assistant", "content": a_low}
                ]
            }

            train_records.append(record_high)
            train_records.append(record_low)
        
        random.shuffle(train_records)
        save_path = os.path.join(output_dir, "train", f"train_{res_type}_{lang1}-{lang2}.jsonl")
        save_jsonl(train_records, save_path)

        test_langs = [l for l in all_languages if l not in (lang1, lang2)]
        test_dir = os.path.join(output_dir, "test", f"{res_type}_{lang1}-{lang2}")
        os.makedirs(test_dir, exist_ok=True)

        for test_lang in test_langs:
            test_records = []
            for item in data:
                d = item["data"]
                q = d[test_lang]["question_2"]
                a = d[test_lang]["answer"]
                ca = d[test_lang]["conflict_answer"]

                test_records.append({
                    "messages": [
                        {"role": "system", "content": f"You are a knowledgeable expert. Please answer the following question concisely in {test_lang}."},
                        {"role": "user", "content": q},
                        {"role": "assistant", "answer": a, "conflict_answer": ca}
                    ]
                })
            
            save_jsonl(test_records, os.path.join(test_dir, f"test_{test_lang}.jsonl"))
    
    with open(os.path.join(output_dir, "selected_pairs.json"), "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, indent=2)

# ----------------------------
# CLI
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--info_path", type=str, default="../../utils/info.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args()

    build_conflict_datasets(
        input_path=args.input_path,
        info_path=args.info_path,
        output_dir=args.output_dir,
        seed=args.seed,
        samples=args.samples
    )