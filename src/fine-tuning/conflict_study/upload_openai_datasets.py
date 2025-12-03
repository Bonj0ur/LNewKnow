# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import argparse
from openai import OpenAI

# ----------------------------
# Utils
# ----------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------
# Batch Upload for Train Files
# ----------------------------
def batch_upload_files(folder_path, output_path, info_path):
    info = load_json(info_path)
    client = OpenAI(api_key=info["apikey"])

    file_ids = {}
    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(".jsonl"):
            continue
        full_path = os.path.join(folder_path, filename)
        try:
            print(f"Uploading: {filename}")
            resp = client.files.create(file=open(full_path, "rb"), purpose="fine-tune")
            file_ids[filename] = resp.id
            print(resp)
            print(f"✅ Uploaded {filename} => File ID: {resp.id}")
        except Exception as e:
            print(f"❌ Failed to upload {filename}: {e}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(file_ids, f, indent=2, ensure_ascii=False)
    print(f"\n🎉 Upload complete. File IDs saved to: {output_path}")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--info_path", type=str, default="../../utils/info.json")
    args = parser.parse_args()

    batch_upload_files(args.folder, args.output, args.info_path)