#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : openchat kit gradio application
"""
            Run:
                # under OpenChatKit/inference from https://github.com/togethercomputer/OpenChatKit
                CUDA_VISIBLE_DEVICES=2,3 python3 gradio_openchatkit.py
            Warn:
                the bigger max_new_tokens the more cuda mem, so be careful
"""

import os
import sys

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(CUR_DIR, "../huggingface_models/GPT-NeoXT-Chat-Base-20B/")

sys.path.append(CUR_DIR)

from loguru import logger
import gradio as gr
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

from bot import ChatModel
from conversation import Conversation


class ConvChat(object):
    """
    Conversation Chat
    """
    def __init__(self,
                 model_name: str,
                 max_new_tokens: int = 256,
                 sample: bool = False,
                 temperature: int = 0.6,
                 top_k: int = 40):
        self.max_new_tokens = max_new_tokens
        self.sample = sample
        self.temperature = temperature
        self.top_k = top_k

        logger.info("Start to init Chat Model")
        self.chat_model = ChatModel(model_name=model_name, gpu_id=0)

        self.conv = Conversation(self.chat_model.human_id, self.chat_model.bot_id)
        logger.info("Initialized Chat Model")

    def run_text(self, input_text: gr.Textbox, state: gr.State):
        self.conv.push_human_turn(input_text)

        output = self.chat_model.do_inference(
            prompt=self.conv.get_raw_prompt(),
            max_new_tokens=self.max_new_tokens,
            do_sample=self.sample,
            temperature=self.temperature,
            top_k=self.top_k
        )
        self.conv.push_model_response(output)
        response = self.conv.get_last_turn()

        state = state + [(input_text, response)]
        return state, state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_PATH, help="model huggingface repo name or local path")
    parser.add_argument("--server_port", type=int, default=7800, help="gradio server port")

    args = parser.parse_args()

    conv_chat = ConvChat(model_name=args.model_name)

    with gr.Blocks(css="OpenChatKit .overflow-y-auto{height:500px}") as gr_chat:
        chatbot = gr.Chatbot(elem_id="chatbot", label="OpenChatKit")
        state = gr.State([])

        with gr.Row():
            with gr.Column(scale=0.8):
                input_text = gr.Textbox(show_label=False,
                                        placeholder="Enter your question").style(container=False)
            with gr.Column(scale=0.2, min_width=0):
                clear_btn = gr.Button("Clear")

        input_text.submit(conv_chat.run_text, [input_text, state], [chatbot, state])
        input_text.submit(lambda: "", None, input_text)

        clear_btn.click(lambda: [], None, chatbot)
        clear_btn.click(lambda: [], None, state)

        gr_chat.launch(
            server_name="0.0.0.0",
            server_port=args.server_port
        )
