import argparse
import json
import time
import torch
import torchvision
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Returns a list of indices for available GPU devices based on memory loads.
def get_available_devices(mem_load_threshold: float = 0.05):
    device_count = torch.cuda.device_count()
    available_devices = []
    for i in range(device_count):
        device = torch.device(f"cuda:{i}")

        mem_used = torch.cuda.memory_allocated(device)
        mem_total = torch.cuda.get_device_properties(device).total_memory
        mem_load = mem_used / mem_total

        # Check if the device is being used 
        if mem_load < mem_load_threshold:
            # If the device is not being used, add it to the list of available GPUs
            available_devices.append(device)

    return available_devices

# Benchmark download, tokenize, load, inference time.
def benchmark(model_dict: dict):

    # Initialize the benchmark results dictionary
    results_dict = {}

    # Check that we have CUDA GPUs available before running the benchmark
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPUs are not available, benchmark not run")
        return results_dict

    # Load the GPU
    available_devices = get_available_devices()

    if len(available_devices) == 0:
        print("ERROR: All CUDA GPUs are being used, benchmark not run")
        return results_dict
    
    device = available_devices[0]

    # Loop through the models to test
    for model_name, model_path in model_dict.items():
        print(f"Testing model: {model_name}")


        # Measure the time it takes to setup the tokenizer
        tokenize_start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(mname)
        tokenize_end_time = time.time()
        tokenize_time = tokenize_end_time - tokenize_start_time

        print(f"Testing model: {model_name} --- tokenize time  = {load_time}")

        # Measure the time it takes to download the model
        download_start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(mname, torch_dtype=torch.float16, torchscript=True, force_download=True)
        download_end_time = time.time()
        download_time = download_end_time - download_start_time

        print(f"Testing model: {model_name} --- download time  = {download_time}")

        # Measure the time it takes to load the model onto the GPU
        load_start_time = time.time()
        model = model.to(device)
        load_end_time = time.time()
        load_time = load_end_time - load_start_time

        print(f"Testing model: {model_name} --- load time      = {load_time}")

        # Measure the time it takes to run inference
        inference_start_time = time.time()
        inputs = tokenizer("Hello, world!", return_tensors="pt").to(device)
        outputs = model(**inputs)
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time

        print(f"Testing model: {model_name} --- inference time = {inference_time}")

        # Add the results to the dictionary
        results_dict[model_name] = {
            "download_time": download_time,
            "tokenize_time": tokenize_time,
            "load_time": load_time,
            "inference_time": inference_time
        }

        # Unload the model
        model = None
        torch.cuda.empty_cache()

    return results_dict

# Define the main function
def main(input_file, output_file):

    # Load the models to test from the input file
    with open(input_file, "r") as f:
        model_dict = json.load(f)

    # Run the benchmark
    results_dict = benchmark(model_dict)

    # Write the results to the output file
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=4)

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Process JSON file')
    parser.add_argument('-i', '--input', required=True, help='input JSON file containing models to be benchmark')
    parser.add_argument('-o', '--output', required=True, help='output JSON file with model benchmark results')

    # Parse the command line arguments
    args = parser.parse_args()

    # Process the data
    main(args.input, args.output)