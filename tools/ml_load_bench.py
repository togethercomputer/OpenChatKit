import argparse
import json
import time
import torch
import torchvision
import os
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM

# Benchmark download, tokenize, load, inference time.
def benchmark(model_dict: dict, device_name: str):

    # Initialize the benchmark results dictionary
    results_dict = {}

    # Check that we have CUDA GPUs available before running the benchmark
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPUs are not available, benchmark not run")
        return results_dict

    device = torch.device(device_name)

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

        tokenizer_load_time = tokenizer_load_end_time - tokenizer_load_start_time
        tokenizer_download_time = tokenizer_download_end_time - tokenizer_download_start_time - tokenizer_load_time

        print(f"Testing model: {model_name} --- tokenizer download time = {tokenizer_download_time:.3} sec")
        print(f"Testing model: {model_name} --- tokenize load time = {tokenizer_load_time:.3} sec")

        # Measure the time it takes to download and load the model into main memory
        model_download_start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, torchscript=True, force_download=True)
        model_download_end_time = time.time()
        
        model = None

        # Measure the time it takes to load the model into main memory
        model_load_to_ram_start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, torchscript=True)
        model_load_to_ram_end_time = time.time()

        model_load_to_ram_time = model_load_to_ram_end_time - model_load_to_ram_start_time
        model_download_time = model_download_end_time - model_download_start_time - model_load_to_ram_time

        print(f"Testing model: {model_name} --- model download time = {model_download_time:.3} sec")
        print(f"Testing model: {model_name} --- model load to RAM time = {model_load_to_ram_time:.3} sec")

        # Measure the time it takes to load the model from main memory to the GPU
        model_xfer_to_gpu_start_time = time.time()
        model = model.to(device)
        model_xfer_to_gpu_end_time = time.time()
        model_xfer_to_gpu_time = model_xfer_to_gpu_end_time - model_xfer_to_gpu_start_time

        print(f"Testing model: {model_name} --- model transfer to GPU time = {model_xfer_to_gpu_time:.3} sec")

        # Measure the time it takes to run inference
        inference_start_time = time.time()
        inputs = tokenizer("Hello, world!", return_tensors="pt").to(device)
        outputs = model(**inputs)
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time

        print(f"Testing model: {model_name} --- inference time = {inference_time:.3} sec")

        total_time = tokenizer_download_time + tokenizer_load_time + model_download_time + model_load_to_ram_time + model_xfer_to_gpu_time + inference_time

        print(f"Testing model: {model_name} --- total time = {total_time:.3} sec")

        # Add the results to the dictionary
        results_dict[model_name] = {
            "tokenizer_download_time": tokenizer_download_time,
            "tokenizer_load_time": tokenizer_load_time,
            "model_download_time": model_download_time,
            "model_load_to_ram_time": model_load_to_ram_time,
            "model_transfer_to_gpu_time": model_xfer_to_gpu_time,
            "inference_time": inference_time,
            "total_time": total_time
        }

        # Unload the model
        model = None
        torch.cuda.empty_cache()

    return results_dict

# Define the main function
def main(input_file, output_file, device_name):

    # Load the models to test from the input file
    with open(input_file, "r") as f:
        model_dict = json.load(f)

    # Run the benchmark
    results_dict = benchmark(model_dict, device_name)

    # Write the results to the output file
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=4)

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Process JSON file')
    parser.add_argument('-i', '--input', required=True, help='input JSON file containing models to be benchmark')
    parser.add_argument('-o', '--output', required=True, help='output JSON file with model benchmark results')
    parser.add_argument('-d', '--device', required=False, default='cuda:0', help='Cuda device name')

    # Parse the command line arguments
    args = parser.parse_args()

    # Process the data
    main(args.input, args.output, args.device)