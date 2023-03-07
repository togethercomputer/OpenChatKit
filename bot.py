import cmd
import json
import readline
import requests

import conversation as convo

endpoint = "https://staging.together.xyz/api/inference"


class OpenChatKitShell(cmd.Cmd):
    intro = "Welcome to OpenChatKit shell.   Type /help or /? to list commands.\n"
    prompt = ">>> "

    def preloop(self):
        self._convo = convo.Conversation('<human>', '<bot>')

    def precmd(self, line):
        if line.startswith('/'):
            return line[1:]
        else:
            return 'say ' + line

    def do_say(self, arg):
        self._convo.push_human_turn(arg)

        response = send_prompt(self._convo.get_raw_prompt())
        model_response = response['output']['choices'][0]['text']
        self._convo.push_model_response(model_response)

        print(self._convo.get_last_turn())

    def do_raw_prompt(self, arg):
        print(self._convo.get_raw_prompt())

    def do_quit(self, arg):
        return True


def send_prompt(prompt):
    res = requests.post(endpoint, json={
        "model": "gpt-neoxt-chat-20b-v0.13-hf",
        "prompt": prompt,
        "top_p": 1,
        "top_k": 50,
        "temperature": 0.6,
        "max_tokens": 128,
    }, headers={
        "Authorization": "Bearer 25243114d5839658949943132330742e15cd830311bf48ab25366e1f6b7452a0",
        "User-Agent": "bot.py"
    })

    return res.json()


if __name__ == '__main__':
    OpenChatKitShell().cmdloop()
