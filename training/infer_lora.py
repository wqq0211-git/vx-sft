#!/usr/bin/env python3
import argparse

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def build_prompt(system_prompt: str, user_message: str, history: str) -> str:
    lines = [f'<|system|>\n{system_prompt}\n']
    if history:
        for chunk in history.split('\n'):
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.startswith('user:'):
                lines.append(f"<|user|>\n{chunk[5:].strip()}\n")
            elif chunk.startswith('assistant:'):
                lines.append(f"<|assistant|>\n{chunk[10:].strip()}\n")
    lines.append(f'<|user|>\n{user_message}\n')
    lines.append('<|assistant|>\n')
    return ''.join(lines)


def clean_output(text: str) -> str:
    text = text.split('<|')[0].strip()
    for token in ['<s>', '</s>']:
        text = text.replace(token, '')
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/lora_qwen3b.yaml')
    parser.add_argument('--adapter-path', required=True)
    parser.add_argument('--message', required=True)
    parser.add_argument('--history', default='')
    parser.add_argument('--max-new-tokens', type=int, default=48)
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config['model_name_or_path'],
        trust_remote_code=True,
        device_map='auto',
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    prompt = build_prompt(config['system_prompt'], args.message, args.history)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.45,
            top_p=0.85,
            repetition_penalty=1.08,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=False)
    print(clean_output(text))


if __name__ == '__main__':
    main()
