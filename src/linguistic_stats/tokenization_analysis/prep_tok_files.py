# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import tiktoken
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from utils import FLORES_200_CODE, TOKENIZERS_BY_TYPE, TRAIN_DATA_LIMIT

def tok_file_exists(dataset, subset, model_type, src_lang, tgt_lang):
    if not subset:
        file_name = f'./datasets/{dataset}/{model_type}/{src_lang}-{tgt_lang}.tok.fast_align'
    else:
        file_name = f'./datasets/{dataset}-{subset}/{model_type}/{src_lang}-{tgt_lang}.tok.fast_align'
    return os.path.isfile(file_name)

def get_tokenizer(model_type):
    model = TOKENIZERS_BY_TYPE[model_type]
    if model.startswith('gpt'):
        enc = tiktoken.encoding_for_model(model)
        return enc, "openai"
    else:
        return AutoTokenizer.from_pretrained(model), "hf"

def get_tokens(dataset, model_type, src_lang, tgt_lang, src_key=None, tgt_key=None):
    if src_key is None:
        src_key = src_lang
    if tgt_key is None:
        tgt_key = tgt_lang
    
    tokenizer, tok_type = get_tokenizer(model_type)
    if tok_type == "hf":
        src_tokens = dataset.map(lambda x: {"tok": tokenizer.convert_ids_to_tokens(tokenizer(x[src_key])["input_ids"], skip_special_tokens=True)})
        tgt_tokens = dataset.map(lambda x: {'tok': tokenizer.convert_ids_to_tokens(tokenizer(x[tgt_key])['input_ids'], skip_special_tokens=True)})
    elif tok_type == "openai":
        src_tokens = dataset.map(lambda x: {"tok": [tokenizer.decode_single_token_bytes(t).decode("utf-8", errors="ignore") for t in tokenizer.encode(x[src_key])]})
        tgt_tokens = dataset.map(lambda x: {"tok": [tokenizer.decode_single_token_bytes(t).decode("utf-8", errors="ignore") for t in tokenizer.encode(x[tgt_key])]})
    return src_tokens, tgt_tokens

def make_tok_file_flores(model_type, src_short, tgt_short, subset="dev"):
    src_tag = FLORES_200_CODE[src_short]
    tgt_tag = FLORES_200_CODE[tgt_short]
    flores = load_dataset(path="../datasets/flores.py", name=f"{src_tag}-{tgt_tag}", trust_remote_code=True)
    dataset = flores[subset]
    return get_tokens(dataset, model_type, src_short, tgt_short, src_key=f"sentence_{src_tag}", tgt_key=f"sentence_{tgt_tag}")

def make_tok_file_multi_cc(model_type, src_lang, tgt_lang):
    src_file = f"../datasets/MultiCCAligned/{src_lang}-{tgt_lang}/MultiCCAligned.{src_lang}-{tgt_lang}.{src_lang}"
    tgt_file = f"../datasets/MultiCCAligned/{src_lang}-{tgt_lang}/MultiCCAligned.{src_lang}-{tgt_lang}.{tgt_lang}"

    if not (os.path.exists(src_file) and os.path.exists(tgt_file)):
        src_file = f"../datasets/MultiCCAligned/{tgt_lang}-{src_lang}/MultiCCAligned.{tgt_lang}-{src_lang}.{src_lang}"
        tgt_file = f"../datasets/MultiCCAligned/{tgt_lang}-{src_lang}/MultiCCAligned.{tgt_lang}-{src_lang}.{tgt_lang}"

    def gen():
        with open(src_file, "r") as srcf, open(tgt_file, "r") as tgtf:
            for i, (srcline, tgtline) in enumerate(zip(srcf, tgtf)):
                if i >= TRAIN_DATA_LIMIT:
                    break
                yield {src_lang: srcline.strip(), tgt_lang: tgtline.strip()}

    dataset = Dataset.from_generator(gen)
    return get_tokens(dataset, model_type, src_lang, tgt_lang)

def make_tok_file_opus(model_type, src_lang_short_code, tgt_lang_short_code, subset="train"):
    if src_lang_short_code < tgt_lang_short_code:
        opus = load_dataset("Helsinki-NLP/opus-100", f"{src_lang_short_code}-{tgt_lang_short_code}", streaming=True, split=subset)
    else:
        opus = load_dataset("Helsinki-NLP/opus-100", f"{tgt_lang_short_code}-{src_lang_short_code}", streaming=True, split=subset)
    dataset = opus.map(lambda x: {src_lang_short_code: x['translation'][src_lang_short_code], tgt_lang_short_code: x['translation'][tgt_lang_short_code]})
    dataset = dataset.take(TRAIN_DATA_LIMIT)
    return get_tokens(dataset, model_type, src_lang_short_code, tgt_lang_short_code)

def write_tok_file(dataset, subset, model_type, src_lang, src_tokens, tgt_lang, tgt_tokens):
    zip_tokens = zip(list(src_tokens), list(tgt_tokens))
    if not subset:
        file_name = f'./datasets/{dataset}/{model_type}/{src_lang}-{tgt_lang}.tok.fast_align'
    else:
        file_name = f'./datasets/{dataset}-{subset}/{model_type}/{src_lang}-{tgt_lang}.tok.fast_align'
    with open(file_name, "w+") as f:
        for src, tgt in zip_tokens:
            src = src["tok"]
            tgt = tgt["tok"]
            f.write(" ".join(src) + " ||| " + " ".join(tgt) + "\n")

def main(args):
    assert (args.src_lang != args.tgt_lang)
    if args.dataset == "flores":
        subset = args.subset or "dev"
    elif args.dataset == "multi-cc":
        subset = args.subset = None
    elif args.dataset == "opus-100":
        subset = args.subset or "train"
    else:
        raise NotImplementedError
    
    out_dir = f"./datasets/{args.dataset}-{args.subset}/{args.model_type}" if subset else f"./datasets/{args.dataset}/{args.model_type}"
    os.makedirs(out_dir, exist_ok=True)

    if tok_file_exists(args.dataset, subset, args.model_type, args.src_lang, args.tgt_lang):
        return

    src_tokens, tgt_tokens = [], []
    if args.dataset == "flores":
        src_tokens, tgt_tokens = make_tok_file_flores(args.model_type, args.src_lang, args.tgt_lang, subset)
    elif args.dataset == "multi-cc":
        src_tokens, tgt_tokens = make_tok_file_multi_cc(args.model_type, args.src_lang, args.tgt_lang)
    elif args.dataset == "opus-100":
        src_tokens, tgt_tokens = make_tok_file_opus(args.model_type, args.src_lang, args.tgt_lang, subset)

    write_tok_file(args.dataset, subset, args.model_type, args.src_lang, src_tokens, args.tgt_lang, tgt_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang", type=str)
    parser.add_argument("--tgt_lang", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--subset", type=str)
    args = parser.parse_args()
    main(args)