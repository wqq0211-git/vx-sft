#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments


def load_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def build_training_sample(messages: List[Dict[str, str]], system_prompt: str) -> Tuple[str, str]:
    if not messages or messages[-1]['role'] != 'assistant':
        raise ValueError('messages must end with an assistant reply')

    prompt_lines = [f"<|system|>\n{system_prompt}\n"]
    for message in messages[:-1]:
        prompt_lines.append(f"<|{message['role']}|>\n{message['content']}\n")
    prompt_lines.append('<|assistant|>\n')
    prompt_text = ''.join(prompt_lines)
    full_text = f"{prompt_text}{messages[-1]['content']}"
    return prompt_text, full_text


def preprocess_row(row: Dict, system_prompt: str) -> Dict:
    prompt_text, full_text = build_training_sample(row['messages'], system_prompt)
    return {'prompt_text': prompt_text, 'text': full_text}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/lora_qwen3b.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = load_dataset(
        'json',
        data_files={
            'train': config['train_file'],
            'validation': config['valid_file'],
        },
    )

    dataset = dataset.map(
        lambda row: preprocess_row(row, config['system_prompt']),
        remove_columns=dataset['train'].column_names,
    )

    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: Dict[str, List[str]]) -> Dict:
        full_tokens = tokenizer(
            batch['text'],
            truncation=True,
            max_length=int(config['max_seq_length']),
            padding='max_length',
        )
        prompt_tokens = tokenizer(
            batch['prompt_text'],
            truncation=True,
            max_length=int(config['max_seq_length']),
            padding='max_length',
        )

        labels = []
        pad_token_id = tokenizer.pad_token_id
        for input_ids, prompt_ids in zip(full_tokens['input_ids'], prompt_tokens['input_ids']):
            prompt_len = sum(1 for token_id in prompt_ids if token_id != pad_token_id)
            row_labels = []
            for index, token_id in enumerate(input_ids):
                if token_id == pad_token_id or index < prompt_len:
                    row_labels.append(-100)
                else:
                    row_labels.append(token_id)
            labels.append(row_labels)

        full_tokens['labels'] = labels
        return full_tokens

    tokenized = dataset.map(tokenize, batched=True, remove_columns=['prompt_text', 'text'])

    quantization_config = None
    if config.get('use_4bit', False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype='bfloat16',
            bnb_4bit_use_double_quant=True,
        )

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    device_map = None if world_size > 1 else {'': 0}

    model = AutoModelForCausalLM.from_pretrained(
        config['model_name_or_path'],
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map=device_map,
    )

    if config.get('use_4bit', False):
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=int(config['lora_r']),
        lora_alpha=int(config['lora_alpha']),
        lora_dropout=float(config['lora_dropout']),
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        per_device_train_batch_size=int(config['per_device_train_batch_size']),
        per_device_eval_batch_size=int(config['per_device_eval_batch_size']),
        gradient_accumulation_steps=int(config['gradient_accumulation_steps']),
        learning_rate=float(config['learning_rate']),
        num_train_epochs=float(config['num_train_epochs']),
        logging_steps=int(config['logging_steps']),
        evaluation_strategy='steps',
        eval_steps=int(config['eval_steps']),
        save_steps=int(config['save_steps']),
        warmup_ratio=float(config['warmup_ratio']),
        bf16=bool(config.get('bf16', False)),
        report_to='none',
        save_total_limit=2,
        load_best_model_at_end=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        data_collator=None,
    )
    trainer.train()
    trainer.save_model(config['output_dir'])
    tokenizer.save_pretrained(config['output_dir'])

    summary = {
        'model_name_or_path': config['model_name_or_path'],
        'train_file': config['train_file'],
        'valid_file': config['valid_file'],
        'output_dir': config['output_dir'],
    }
    summary_path = Path(config['output_dir']) / 'training_run_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
