# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import argparse
from utils import MODEL_PATH
from types import MethodType
from vllm import LLM, SamplingParams

os.environ["VLLM_USE_V1"] = "0"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",type=str)
    parser.add_argument("--model_name",type=str)
    parser.add_argument("--enforce_eager",action="store_true")
    parser.add_argument("--max_model_len",type=int,default=2048)
    parser.add_argument("--input_dir",type=str,default="./outputs/token_ids/")
    parser.add_argument("--save_dir",type=str,default="./outputs/activations/")
    args = parser.parse_args()

    model = LLM(
        model=MODEL_PATH[args.model_name],
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len
    )

    max_length = model.llm_engine.model_config.max_model_len
    num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
    intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size
    activation_size = model.llm_engine.model_config.hf_config.intermediate_size // 2

    over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
    activation_sums = torch.zeros(num_layers, intermediate_size, dtype=torch.float32).to('cuda')
    activation_counts = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')

    def factory(idx):
        def model_forward(self, x):
            gate_up, _ = self.gate_up_proj(x)
            i = gate_up.size(-1)
            if gate_up.dim() == 3:
                gate_up[:, :, : i // 2] = torch.nn.SiLU()(gate_up[:, :, : i // 2])
                activation = gate_up[:, :, : i // 2].float()
                over_zero[idx, :] += (activation > 0).sum(dim=(0, 1))
                activation_sums[idx, :] += activation.sum(dim=(0, 1))
                activation_counts[idx, :] += activation.size(0) * activation.size(1)
                x = gate_up[:, :, : i // 2] * gate_up[:, :, i // 2 :]
            elif gate_up.dim() == 2:
                gate_up[:, : i // 2] = torch.nn.SiLU()(gate_up[:, : i // 2])
                activation = gate_up[:, : i // 2].float()
                over_zero[idx, :] += (activation > 0).sum(dim=0)
                activation_sums[idx, :] += activation.sum(dim=0)
                activation_counts[idx, :] += activation.size(0)
                x = gate_up[:, : i // 2] * gate_up[:, i // 2 :]
            else:
                raise ValueError(f"Unexpected gate_up shape: {gate_up.shape}")
            x, _ = self.down_proj(x)
            return x
        return model_forward

    for i in range(num_layers):
        obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
        obj.forward = MethodType(factory(i), obj)
    
    id_path = os.path.join(args.input_dir,f"id.{args.lang}.flores.{args.model_name}")
    ids = torch.load(id_path)
    l = ids.size(0)
    l = l // max_length * max_length
    input_ids = ids[:l].reshape(-1, max_length)
    output = model.generate(prompt_token_ids=input_ids.tolist(), sampling_params=SamplingParams(max_tokens=1))
    average_activations = activation_sums / activation_counts.float()

    output = dict(
        n=l, 
        over_zero=over_zero.to('cpu'),
        average_activations=average_activations.to('cpu'),
        activation_counts=activation_counts.to('cpu')
    )

    save_path = os.path.join(args.save_dir,f"activation.{args.lang}.flores.{args.model_name}")
    torch.save(output, save_path)