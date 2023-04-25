import argparse
import json
import time
import torch
import torchvision
import os
import re
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM

# Benchmark download, tokenize, load, inference time.
def benchmark(model_dict: dict, device_name: str, repeat_infer: int):

    # Initialize the benchmark results dictionary
    results_dict = {}

    # Check that we have CUDA GPUs available before running the benchmark
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPUs are not available, benchmark not run")
        return results_dict

    device = torch.device(device_name)

    process = psutil.Process()

    print(f'Using device {device}')

    # Loop through the models to test
    for model_name, model_path in model_dict.items():
        # purge unused cached memory
        torch.cuda.empty_cache()

        print(f"Testing model: {model_name}")

        # Measure the time it takes to download the tokenizer data and load the tokenizer
        tokenizer_download_start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_path, force_download=True)
        tokenizer_download_end_time = time.time()

        tokenizer = None

        # Measure the time it takes to  load the tokenizer
        tokenizer_load_start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer_load_end_time = time.time()

        tokenizer_load_sec = tokenizer_load_end_time - tokenizer_load_start_time
        tokenizer_download_sec = tokenizer_download_end_time - tokenizer_download_start_time - tokenizer_load_sec

        print(f"Testing model: {model_name} --- tokenizer download time = {tokenizer_download_sec:.3} sec")
        print(f"Testing model: {model_name} --- tokenize load time = {tokenizer_load_sec:.3} sec")

        # Measure the time it takes to download and load the model into main memory
        model_download_start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, torchscript=True, force_download=True)
        model_download_end_time = time.time()
        
        model = None

        # Measure the time it takes to load the model into main memory
        memory_used_main_start = process.memory_info().rss
        model_load_to_ram_start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, torchscript=True)
        model_load_to_ram_end_time = time.time()
        memory_used_main_end = process.memory_info().rss

        model_load_to_ram_sec = model_load_to_ram_end_time - model_load_to_ram_start_time
        model_download_sec = model_download_end_time - model_download_start_time - model_load_to_ram_sec
        model_main_memory_bytes = memory_used_main_end - memory_used_main_start

        print(f"Testing model: {model_name} --- model download time = {model_download_sec:.3} sec")
        print(f"Testing model: {model_name} --- model load to RAM time = {model_load_to_ram_sec:.3} sec")
        print(f"Testing model: {model_name} --- model main memory size = {model_main_memory_bytes} bytes")

        # Measure the time it takes to load the model from main memory to the GPU
        gpu_memory_start = torch.cuda.memory_allocated(device)
        model_xfer_to_gpu_start_time = time.time()
        model = model.to(device)
        model_xfer_to_gpu_end_time = time.time()
        gpu_memory_end = torch.cuda.memory_allocated(device)

        model_xfer_to_gpu_sec = model_xfer_to_gpu_end_time - model_xfer_to_gpu_start_time
        model_gpu_memory_bytes = gpu_memory_end - gpu_memory_start

        print(f"Testing model: {model_name} --- model transfer to GPU time = {model_xfer_to_gpu_sec:.3} sec")
        print(f"Testing model: {model_name} --- model GPU memory size = {model_gpu_memory_bytes} bytes")

        # Measure the time it takes to run inference from a cold start
        inference_start_time = time.time()
        inputs = tokenizer("Hello, world!", return_tensors="pt").to(device)
        outputs = model(**inputs)
        inference_end_time = time.time()
        inference_sec = inference_end_time - inference_start_time

        print(f"Testing model: {model_name} --- inference time = {inference_sec:.3} sec")

        # Measure the time it takes to run inference from a cold start
        inference_warm_start_time = time.time()
        for i in range(0, repeat_infer):
            inputs = tokenizer("Hello, world!", return_tensors="pt").to(device)
            outputs = model(**inputs)
        inference_warm_end_time = time.time()
        inference_warm_sec = (inference_warm_end_time - inference_warm_start_time) / float(repeat_infer)

        print(f"Testing model: {model_name} --- inference warm time = {inference_warm_sec:.3} sec")

        total_sec = tokenizer_download_sec + tokenizer_load_sec + model_download_sec + model_load_to_ram_sec + model_xfer_to_gpu_sec + inference_sec

        print(f"Testing model: {model_name} --- total time = {total_sec:.3} sec")

        # Add the results to the dictionary
        results_dict[model_name] = {
            "tokenizer_download_sec": tokenizer_download_sec,
            "tokenizer_load_sec": tokenizer_load_sec,
            "model_download_sec": model_download_sec,
            "model_load_to_ram_sec": model_load_to_ram_sec,
            "model_main_memory_MB": float(model_main_memory_bytes) / 1000000.0,
            "model_transfer_to_gpu_sec": model_xfer_to_gpu_sec,
            "model_gpu_memory_MB": float(model_gpu_memory_bytes) / 1000000.0,
            "inference_sec": inference_sec,
            "inference_warm_sec": inference_warm_sec,
            "total_sec": total_sec
        }

        # Unload the model
        model = None
        torch.cuda.empty_cache()

    return results_dict

# Define the main function
def main(input_file: str, output_file: str, device_name: str, repeat_infer: int):

    # Load the models to test from the input JSON file
    with open(input_file, "r") as f:
        model_dict = json.load(f)

    # Run the benchmark
    results_dict = benchmark(model_dict, device_name, repeat_infer)

    # Write the results to the JSON output file
    # use a regular expression to apply formatting to floatin point
    json_data = re.sub('"(.*?)":\s*(0\.0*\d{2}|\d+\.\d{2})\d*(,?\n)', '"\\1": \\2\\3',  json.dumps(results_dict, indent=4))
    with open(output_file, 'w') as f:
        f.write(json_data)

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Benchmark downloading, loading, and running an inferernce for a set of ML models.')
    parser.add_argument('-i', '--input', required=True, help='Input JSON file containing models to be benchmark')
    parser.add_argument('-o', '--output', required=True, help='Output JSON file with model benchmark results')
    parser.add_argument('-d', '--device', required=False, default='cuda:0', help='Cuda device name, e.g. "cuda:0"')
    parser.add_argument('-r', '--repeat-infer', required=False, default=30, help='Repeat inferrence for warm timings')

    # Parse the command line arguments
    args = parser.parse_args()

    # Process the data
    main(args.input, args.output, args.device, max(args.repeat_infer, 1))