#!/usr/bin/env python3
import argparse

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatModel:
    def __init__(self, base_model_path: str, adapter_path: str | None, system_prompt: str):
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map='auto',
        )
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.eval()

    def build_prompt(self, history, message, style_mode):
        if style_mode == '更像微信聊天':
            extra = '回复尽量口语化、直接、自然，允许简短，但不要只回标签。'
        else:
            extra = '回复尽量自然稳妥，优先清楚表达，不要续写多轮对话。'
        lines = [f'<|system|>\n{self.system_prompt}\n{extra}\n']
        for user, assistant in history:
            lines.append(f'<|user|>\n{user}\n')
            lines.append(f'<|assistant|>\n{assistant}\n')
        lines.append(f'<|user|>\n{message}\n<|assistant|>\n')
        return ''.join(lines)

    def generate(self, history, message, temperature, top_p, max_new_tokens, style_mode):
        prompt = self.build_prompt(history, message, style_mode)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.12,
                no_repeat_ngram_size=4,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        text = text.split('<|')[0].replace('<s>', '').replace('</s>', '').strip()
        return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--adapter-path', default='')
    parser.add_argument('--system-prompt', default='请使用自然、口语化、简洁的聊天风格回复，避免过度书面化。')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()

    adapter_path = args.adapter_path or None
    chat_model = ChatModel(args.base_model, adapter_path, args.system_prompt)

    def respond(message, history, temperature, top_p, max_new_tokens, style_mode):
        history = history or []
        reply = chat_model.generate(history, message, temperature, top_p, int(max_new_tokens), style_mode)
        history.append((message, reply))
        return history, history

    def clear_history():
        return [], []

    with gr.Blocks(title='微信聊天 风格聊天') as demo:
        gr.Markdown('# 微信聊天 风格聊天 UI')
        chatbot = gr.Chatbot(height=520)
        state = gr.State([])
        with gr.Row():
            message = gr.Textbox(label='输入消息', scale=6)
            send = gr.Button('发送', scale=1)
        with gr.Row():
            clear = gr.Button('清空上下文')
            style_mode = gr.Radio(['更像微信聊天', '更稳妥'], value='更像微信聊天', label='模式')
        with gr.Row():
            temperature = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label='temperature')
            top_p = gr.Slider(0.5, 1.0, value=0.85, step=0.01, label='top_p')
            max_new_tokens = gr.Slider(16, 128, value=48, step=8, label='max_new_tokens')

        send.click(
            respond,
            inputs=[message, state, temperature, top_p, max_new_tokens, style_mode],
            outputs=[chatbot, state],
        )
        clear.click(clear_history, outputs=[chatbot, state])

    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == '__main__':
    main()
