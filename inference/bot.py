import os
import sys

INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))

# TODO: PYTHONPATH hacks are never a good idea. clean this up later
sys.path.append(os.path.join(INFERENCE_DIR, '..'))

import cmd
import torch
import argparse
import conversation as convo
import retrieval.wikipedia as wp
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import infer_auto_device_map, init_empty_weights


class ChatModel:
    human_id = "<human>"
    bot_id = "<bot>"

    def __init__(self, model_name, gpu_id, max_memory):
        device = torch.device('cuda', gpu_id)   # TODO: allow sending to cpu

        # recommended default for devices with > 40 GB vRAM
        # load model onto one device
        if max_memory is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto")
            self._model.to(device)
        # load the model with the given max_memory config (for devices with insufficient vRAM or multi-gpu)
        else:
            config = AutoConfig.from_pretrained(model_name)
            # load empty weights
            with init_empty_weights():
                model_from_conf = AutoModelForCausalLM.from_config(config)

            model_from_conf.tie_weights()

            # create a device_map from max_memory
            device_map = infer_auto_device_map(
                model_from_conf,
                max_memory=max_memory,
                no_split_module_classes=["GPTNeoXLayer"],
                dtype="float16"
            )
            # load the model with the above device_map
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                offload_folder="offload",  # optional offload-to-disk overflow directory (auto-created)
                offload_state_dict=True,
                torch_dtype=torch.float16
            )
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
            top_k=top_k,
            pad_token_id=self._tokenizer.eos_token_id
        )
        output = self._tokenizer.batch_decode(outputs)[0]

        # remove the context from the output
        output = output[len(prompt):]

        return output


class OpenChatKitShell(cmd.Cmd):
    intro = "Welcome to OpenChatKit shell.   Type /help or /? to list commands.\n"
    prompt = ">>> "

    def __init__(self, gpu_id, model_name_or_path, max_tokens, sample, temperature, top_k, retrieval, max_memory):
        super().__init__()
        self._gpu_id = int(gpu_id)
        self._model_name_or_path = model_name_or_path
        self._max_tokens = max_tokens
        self._sample = sample
        self._temperature = temperature
        self._top_k = top_k
        self._retrieval = retrieval
        self._max_memory = max_memory

    def preloop(self):
        print(f"Loading {self._model_name_or_path} to cuda:{self._gpu_id}...")
        self._model = ChatModel(self._model_name_or_path, self._gpu_id, self._max_memory)

        if self._retrieval:
            print(f"Loading retrieval index...")
            self._index = wp.WikipediaIndex()

        self._convo = convo.Conversation(
            self._model.human_id, self._model.bot_id)

    def precmd(self, line):
        if line.startswith('/'):
            return line[1:]
        else:
            return 'say ' + line

    def do_say(self, arg):
        if self._retrieval:
            results = self._index.search(arg)
            if len(results) > 0:
                self._convo.push_context_turn(results[0])

        self._convo.push_human_turn(arg)

        output = self._model.do_inference(
            self._convo.get_raw_prompt(),
            self._max_tokens,
            self._sample,
            self._temperature,
            self._top_k
        )

        self._convo.push_model_response(output)

        print(self._convo.get_last_turn())

    def do_raw_say(self, arg):
        output = self._model.do_inference(
            arg,
            self._max_tokens,
            self._sample,
            self._temperature,
            self._top_k
        )

        print(output)

    def do_raw_prompt(self, arg):
        print(self._convo.get_raw_prompt())

    def do_reset(self, arg):
        self._convo = convo.Conversation(
            self._model.human_id, self._model.bot_id)

    def do_hyperparameters(self, arg):
        print(
            f"Hyperparameters:\n"
            f"  max_tokens: {self._max_tokens}\n"
            f"  sample: {self._sample}\n"
            f"  temperature: {self._temperature}\n"
            f"  top_k: {self._top_k}"
        )

    def do_quit(self, arg):
        return True


def main():
    parser = argparse.ArgumentParser(
        description='test harness for OpenChatKit')

    parser.add_argument(
        '--gpu-id',
        default=0,
        help='the ID of the GPU to run on'
    )
    parser.add_argument(
        '--model',
        default=f"{INFERENCE_DIR}/../huggingface_models/GPT-NeoXT-Chat-Base-20B",
        help='the ID of the GPU to run on'
    )
    parser.add_argument(
        '--max-tokens',
        default=128,
        help='the maximum number of tokens to generate'
    )
    parser.add_argument(
        '--sample',
        default=True,
        action='store_true',
        help='indicates whether to sample'
    )
    parser.add_argument(
        '--temperature',
        default=0.6,
        help='temperature for the LM'
    )
    parser.add_argument(
        '--top-k',
        default=40,
        help='top-k for the LM'
    )
    parser.add_argument(
        '--retrieval',
        default=False,
        action='store_true',
        help='augment queries with context from the retrieval index'
    )
    parser.add_argument(
        '-g',
        '--gpu-vram',
        action='store',
        help='Max vRAM to allocate per GPU',
        nargs='+',
        required=False,
    )
    parser.add_argument(
        '-r',
        '--cpu-ram',
        default=None,
        type=int,
        help='Max CPU RAM to allocate',
        required=False
    )
    args = parser.parse_args()

    # set max_memory dictionary if given
    if args.gpu_vram is None:
        max_memory = None
    else:
        max_memory = {}
        for i in range(len(args.gpu_vram)):
            # assign CUDA ID as label and XGiB as value
            max_memory[int(args.gpu_vram[i].split(':')[0])] = f"{args.gpu_vram[i].split(':')[1]}GiB"

        if args.cpu_ram is not None:
            # add cpu to max-memory if given
            max_memory['cpu'] = f"{int(args.cpu_ram)}GiB"

    OpenChatKitShell(
        args.gpu_id,
        args.model,
        args.max_tokens,
        args.sample,
        args.temperature,
        args.top_k,
        args.retrieval,
        max_memory
    ).cmdloop()


if __name__ == '__main__':
    main()
