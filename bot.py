import cmd
import readline

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import conversation as convo


class ChatModel:
    human_id = "<human>"
    bot_id = "<bot>"

    def __init__(self, model_name, gpu_id):
        device = torch.device('cuda', gpu_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_name).half()
        self._model.to(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def do_inference(self, prompt, max_new_tokens, do_sample, temperature, top_k):
        inputs = (
            self._tokenizer(prompt, return_tensors='pt')
            .to(self._model.device)
        )
        outputs = self._model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=do_sample, 
            temperature=temperature, 
            top_k=top_k
        )
        output = self._tokenizer.batch_decode(outputs)[0]

        # remove the context from the output
        output = output[len(prompt):]

        return output


class OpenChatKitShell(cmd.Cmd):
    intro = "Welcome to OpenChatKit shell.   Type /help or /? to list commands.\n"
    prompt = ">>> "

    gpu_id = 1
    model_name_or_path = './training/GPT-NeoX-fine-tuned'
    max_tokens = 128
    sample = True
    temperature = 0.6
    top_k = 40

    def preloop(self):
        print(f"Loading {self.model_name_or_path} to cuda:{self.gpu_id}...")

        self._model = ChatModel(self.model_name_or_path, self.gpu_id)
        self._convo = convo.Conversation(self._model.human_id, self._model.bot_id)

    def precmd(self, line):
        if line.startswith('/'):
            return line[1:]
        else:
            return 'say ' + line

    def do_say(self, arg):
        self._convo.push_human_turn(arg)

        output = self._model.do_inference(
            self._convo.get_raw_prompt(), 
            self.max_tokens, 
            self.sample, 
            self.temperature, 
            self.top_k
        )

        self._convo.push_model_response(output)

        print(self._convo.get_last_turn())

    def do_raw_say(self, arg):
        output = self._model.do_inference(
            arg,
            self.max_tokens,
            self.sample,
            self.temperature,
            self.top_k
        )

        print(output)

    def do_raw_prompt(self, arg):
        print(self._convo.get_raw_prompt())

    def do_reset(self, arg):
        self._convo = convo.Conversation(self._model.human_id, self._model.bot_id)

    def do_quit(self, arg):
        return True


if __name__ == '__main__':
    OpenChatKitShell().cmdloop()
