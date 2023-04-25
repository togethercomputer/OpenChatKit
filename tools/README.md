# OpenChatKit Tools

## convert_to_hf_gptneox.py

## ml_load_benchmark.py

The commands to run the model load benchmark tool is:
```shell
$ python3 model_load_benchmark.py -i benchmark_input.json -o benchmark_results.json -d cuda:0
```

usage: model_load_benchmark.py [-h] -i INPUT -o OUTPUT [-d DEVICE] [-r REPEAT_INFER]

Benchmark downloading, loading, and running an inferernce for a set of ML models.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input JSON file containing models to be benchmark
  -o OUTPUT, --output OUTPUT
                        Output JSON file with model benchmark results
  -d DEVICE, --device DEVICE
                        Cuda device name, e.g. "cuda:0"
  -r REPEAT_INFER, --repeat-infer REPEAT_INFER
                        Repeat inferrence for warm timings

The input file is a JSON file with the names and paths of the models to be tested. For example:
```JSON
{
    "GPT-NeoXT-Chat-Base-20B": "togethercomputer/GPT-NeoXT-Chat-Base-20B",
    "Pythia-Chat-Base-7B": "togethercomputer/Pythia-Chat-Base-7B",
    "GPT-JT-Moderation-6B": "togethercomputer/GPT-JT-Moderation-6B",
    "GPT-JT-6B-v1": "togethercomputer/GPT-JT-6B-v1",
    "GPT-JT-6B-v0": "togethercomputer/GPT-JT-6B-v0"
}
```

The output is a json file with the timings for:
1. tokenizer download time in seconds -- `tokenizer_download_sec`
2. tokenizer load time in seconds -- `tokenizer_load_sec`
3. model download time -- `model_download_sec`
5. model load to RAM time -- `model_load_to_ram_sec`
6. model transfer to GPU time -- `model_transfer_to_gpu_sec`
7. inference time (input is "hello, world!") -- `inference_sec`
8. total time (sum of all the above) -- `total_sec`
9. inference time from a warm start (the average of running inference `REPEAT_INFER` times) -- `inference_warm_sec`
10. model main memory footprint in MB -- `model_main_memory_MB`
11. model GPU memory footprint in MB -- `model_gpu_memory_MB`

An example of the output is:
```JSON
{
    "GPT-JT-6B-v1": {
        "tokenizer_download_sec": 1.5241949558258057,
        "tokenizer_load_sec": 0.10464024543762207,
        "model_download_sec": 124.70549297332764,
        "model_load_to_ram_sec": 127.81262159347534,
        "model_main_memory_MB": 12297.105408,
        "model_transfer_to_gpu_sec": 3.2990105152130127,
        "model_gpu_memory_MB": 12219.744768,
        "inference_sec": 0.9375214576721191,
        "inference_warm_sec": 0.047981548309326175,
        "total_sec": 258.38348174095154
    }
}
```