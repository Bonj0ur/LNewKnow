# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import tiktoken
import argparse
import numpy as np
from tqdm import tqdm
from itertools import combinations
from transformers import AutoTokenizer
from utils import UNK_TOKEN, TOKENIZERS_BY_TYPE
from scipy.spatial.distance import jensenshannon
from collections import defaultdict, OrderedDict

def get_tokenizer(model_type):
    model = TOKENIZERS_BY_TYPE[model_type]
    if model.startswith("gpt"):
        return tiktoken.encoding_for_model(model), "openai"
    else:
        return AutoTokenizer.from_pretrained(model), "hf"

def get_vocab_map(tokenizer, tok_type):
    if tok_type == "openai":
        vocab = {}
        for b, idx in tokenizer._mergeable_ranks.items():
            try:
                s = tokenizer.decode([idx])
            except Exception:
                s = b.decode("utf-8", errors="replace")
            vocab[s] = idx
        for s, idx in tokenizer._special_tokens.items():
            vocab[s] = idx
        return vocab
    else:
        return tokenizer.get_vocab()

def batch_iter(iterator, batch_size):
    buf = []
    for item in iterator:
        buf.append(item)
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:
        yield buf

def encode_lines(tokenizer, tok_type, lines, hf_add_special_tokens):
    if tok_type == "openai":
        return [tokenizer.encode(s, disallowed_special=()) for s in lines]
    else:
        out = tokenizer(
            list(lines),
            add_special_tokens=hf_add_special_tokens,
            return_attention_mask=False,
            return_token_type_ids=False
        )
        return out["input_ids"]

def id_to_token_str(tokenizer, tok_type, token_id):
    if tok_type == "openai":
        return tokenizer.decode([token_id])
    else:
        return tokenizer.convert_ids_to_tokens([token_id])[0]

def save_token_frequency(tokens_with_freq, decoded_tokens_with_freq, out_path, name):
    os.makedirs(out_path, exist_ok=True)
    for save_name, save_object in [
        (f"{name}.json", tokens_with_freq),
        (f"{name}_decoded.json", decoded_tokens_with_freq),
    ]:
        if not save_object:
            continue
        path = os.path.join(out_path, save_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(OrderedDict(save_object), f, indent=2, ensure_ascii=False)

def compute_frequencies(data_list, tok_type, tokenizer, name, output_path, batch_size, hf_add_special_tokens):
    vocab = get_vocab_map(tokenizer, tok_type)
    id_counter = {tid: 0 for tid in vocab.values()}
    for data_path in data_list:
        with open(data_path, "r", encoding="utf-8") as f:
            for lines in tqdm(batch_iter(map(lambda s: s.rstrip("\n"), f), batch_size), desc=f"Counting tokens in {os.path.basename(data_path)}"):
                for token_ids in encode_lines(tokenizer, tok_type, lines, hf_add_special_tokens):
                    for tid in token_ids:
                        id_counter[tid] = id_counter.get(tid, 0) + 1
    tokens_with_freq = sorted(id_counter.items(), key=lambda x: x[1], reverse=True)
    decoded_tokens_with_freq = [(id_to_token_str(tokenizer, tok_type, tid), freq) for tid, freq in tokens_with_freq]
    save_token_frequency(tokens_with_freq, decoded_tokens_with_freq, output_path, name)

def compute_number_of_characters(lang2data):
    number_of_characters = defaultdict(int)
    for lang, data_paths in lang2data.items():
        for data_path in data_paths:
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    number_of_characters[lang] += len(line)
                    number_of_characters["All"] += len(line)
    return number_of_characters

def load_frequencies_decoded(tokenizer_dir, languages):
    freqs_by_lang = {}
    for lang in languages:
        p = os.path.join(tokenizer_dir, f"token_freq_{lang}_decoded.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                freqs_by_lang[lang] = OrderedDict(json.load(f))
    p_all = os.path.join(tokenizer_dir, "token_frequencies_decoded.json")
    if os.path.exists(p_all):
        with open(p_all, "r", encoding="utf-8") as f:
            freqs_by_lang["All"] = OrderedDict(json.load(f))
    return freqs_by_lang

def distribution_from_frequencies(freq_map):
    total = float(np.sum(list(freq_map.values())))
    if total == 0:
        return OrderedDict((k, 0.0) for k in freq_map.keys())
    return OrderedDict((k, v / total) for k, v in freq_map.items())

def align_vocab_and_build_distributions(freqs_by_lang):
    tokens_union = []
    seen = set()
    for freqs in freqs_by_lang.values():
        for tok in freqs.keys():
            if tok not in seen:
                seen.add(tok)
                tokens_union.append(tok)
    tokens_union.sort()
    distributions = {}
    raw_freqs_aligned = {}
    total_tokens = {}
    for lang, freqs in freqs_by_lang.items():
        aligned = OrderedDict((tok, freqs.get(tok, 0)) for tok in tokens_union)
        raw_freqs_aligned[lang] = aligned
        total = int(np.sum(list(aligned.values())))
        total_tokens[lang] = total
        dist = distribution_from_frequencies(aligned)
        distributions[lang] = np.array(list(dist.values()), dtype=float)
    return tokens_union, distributions, raw_freqs_aligned, total_tokens

def compute_jsd(p1, p2, base=2.0):
    return float(jensenshannon(p1, p2, base=base) ** 2)

def compute_average_rank(probabilities):
    sorted_p = np.sort(probabilities)[::-1]
    r_e = float(np.sum(sorted_p * np.arange(len(probabilities))))
    return r_e

def get_properties(languages, out_dir, number_of_characters, unk_token):
    freqs_by_lang = load_frequencies_decoded(out_dir, languages)
    langs = sorted(set(languages)) + (["All"] if "All" in freqs_by_lang else [])
    tokens_union, distributions, raw_freqs, total_tokens = align_vocab_and_build_distributions({k: v for k, v in freqs_by_lang.items() if k in langs})
    props = {}
    props["JSD"] = {}
    for l1, l2 in combinations(langs, 2):
        props["JSD"][f"{l1}-{l2}"] = compute_jsd(distributions[l1], distributions[l2])
    props["Average Rank"] = {lang: compute_average_rank(distributions[lang]) for lang in langs}
    props["Characters per Token"] = {lang: float(number_of_characters.get(lang, 0)) / float(total_tokens.get(lang, 1)) for lang in langs}
    props["Coverage"] = {}
    for lang in langs:
        unk_freq = raw_freqs[lang].get(unk_token, None)
        if unk_freq is None:
            props["Coverage"][lang] = None
        else:
            total = total_tokens.get(lang, 0) or 1
            props["Coverage"][lang] = 1.0 - (float(unk_freq) / float(total))
    props["Union Vocab Size"] = len(tokens_union)
    return props

def main(args):
    tokenizer, tok_type = get_tokenizer(args.model_name)
    lang2data = defaultdict(list)
    for lang, data_path in zip(args.languages, args.data_list):
        lang2data[lang].append(data_path)
    os.makedirs(args.output_path, exist_ok=True)
    for lang, data_paths in lang2data.items():
        out_file = os.path.join(args.output_path, f"token_freq_{lang}_decoded.json")
        if not os.path.exists(out_file):
            compute_frequencies(data_paths, tok_type=tok_type, tokenizer=tokenizer, name=f"token_freq_{lang}", output_path=args.output_path, batch_size=args.batch_size, hf_add_special_tokens=args.hf_add_special_tokens)
    all_out = os.path.join(args.output_path, "token_frequencies_decoded.json")
    if not os.path.exists(all_out):
        compute_frequencies(args.data_list,tok_type=tok_type, tokenizer=tokenizer, name="token_frequencies", output_path=args.output_path, batch_size=args.batch_size, hf_add_special_tokens=args.hf_add_special_tokens)
    number_of_characters = compute_number_of_characters(lang2data)
    props = get_properties(args.languages, args.output_path, number_of_characters, args.unk_token)
    out_file = os.path.join(args.output_path, "tokenizer_properties.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(props, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved properties to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_list", nargs="+", required=True)
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--unk_token", type=str, default=UNK_TOKEN)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--hf_add_special_tokens", action="store_true")
    args = parser.parse_args()
    main(args)