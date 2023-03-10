import re
import time

MEANINGLESS_WORDS = ['<pad>', '</s>', '<|endoftext|>']
PRE_PROMPT = """\
Current Date: {}
Current Time: {}

"""

def clean_response(response):
    for word in MEANINGLESS_WORDS:
        response = response.replace(word, "")
    response = response.strip("\n")
    return response

class Conversation:
    def __init__(self, human_id, bot_id):
        cur_date = time.strftime('%Y-%m-%d')
        cur_time = time.strftime('%H:%M:%S %p %Z')

        self._human_id = human_id
        self._bot_id = bot_id
        self._prompt = PRE_PROMPT.format(cur_date, cur_time)

    def push_context_turn(self, context):
        # for now, context is represented as a human turn
        self._prompt += f"{self._human_id}: {context}\n"

    def push_human_turn(self, query):
        self._prompt += f"{self._human_id}: {query}\n"
        self._prompt += f"{self._bot_id}:"

    def push_model_response(self, response):
        has_finished = self._human_id in response
        bot_turn = response.split(f"{self._human_id}:")[0]
        bot_turn = clean_response(bot_turn)
        # if it is truncated, then append "..." to the end of the response
        if not has_finished:
            bot_turn += "..."

        self._prompt += f"{bot_turn}\n"

    def get_last_turn(self):
        human_tag = f"{self._human_id}:"
        bot_tag = f"{self._bot_id}:"
        turns = re.split(f"({human_tag}|{bot_tag})\W?", self._prompt)
        return turns[-1]

    def get_raw_prompt(self):
        return self._prompt

    @classmethod
    def from_raw_prompt(cls, value):
        self._prompt = value
