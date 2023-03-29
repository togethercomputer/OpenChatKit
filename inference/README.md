# OpenChatKit Inference
This directory contains code for OpenChatKit's inference.

## Arguments
- `--gpu-id`: Primary GPU device to load inputs onto for inference. Default: `0`
- `--model`: name/path of the model. Default = `../huggingface_models/GPT-NeoXT-Chat-Base-20B`
- `--max-tokens`: the maximum number of tokens to generate. Default: `128`
- `--sample`: indicates whether to sample. Default: `True`
- `--temperature`: temperature for the LM. Default: `0.6`
- `--top-k`: top-k for the LM. Default: `40`
- `--retrieval`: augment queries with context from the retrieval index. Default `False`
- `-g` `--gpu-vram`: GPU ID and VRAM to allocate to loading the model, separated by a `:` in the format `ID:RAM` where ID is the CUDA ID and RAM is in GiB. `gpu-id` must be present in this list to avoid errors. Accepts multiple values, for example, `-g ID_0:RAM_0 ID_1:RAM_1 ID_N:RAM_N`
- `-r` `--cpu-ram`: CPU RAM overflow allocation for loading the model. Optional, and only used if the model does not fit onto the GPUs given.

## Hardware requirements for inference
The GPT-NeoXT-Chat-Base-20B model requires at least 41GB of free VRAM. Used VRAM also goes up by ~100-200 MB per prompt. 

- A **minimum of 80 GB is recommended** 

- A **minimum of 48 GB in VRAM is recommended** for fast responses.

If you'd like to run inference on a GPU with <48 GB VRAM, refer to this section on [running on consumer hardware](#running-on-consumer-hardware).

By default, inference uses only CUDA Device 0.

**NOTE: Inference currently requires at least 1x GPU.**

## Running on multiple GPUs
Add the argument 

```-g ID0:MAX_VRAM ID1:MAX_VRAM ID2:MAX_VRAM ...``` 

where IDx is the CUDA ID of the device and MAX_VRAM is the amount of VRAM you'd like to allocate to the device.

For example, if you are running this on 4x 48 GB GPUs and want to distribute the model across all devices, add ```-g 0:10 1:12 2:12 3:12 4:12```. In this example, the first device gets loaded to a max of 10 GiB while the others are loaded with a max of 12 GiB.

How it works: The model fills up the max available VRAM on the first device passed and then overflows into the next until the whole model is loaded.

**IMPORTANT: This MAX_VRAM is only for loading the model. It does not account for the additional inputs that are added to the device. It is recommended to set the MAX_VRAM to be at least 1 or 2 GiB less than the max available VRAM on each device, and at least 3GiB less than the max available VRAM on the primary device (set by `gpu-id` default=0).**

**Decrease MAX_VRAM if you run into CUDA OOM. This happens because each input takes up additional space on the device.**

**NOTE: Total MAX_VRAM across all devices must be > size of the model in GB. If not, `bot.py` automatically offloads the rest of the model to RAM and disk. It will use up all available RAM. To allocate a specified amount of RAM: [refer to this section on running on consumer hardware](#running-on-consumer-hardware).**

## Running on specific GPUs
If you have multiple GPUs but would only like to use a specific device(s), [use the same steps as in this section on running on multiple devices](#running-on-multiple-gpus) and only specify the devices you'd like to use. 

Also, if needed, add the argument `--gpu-id ID` where ID is the CUDA ID of the device you'd like to make the primary device. NOTE: The device specified in `--gpu-id` must be present as one of the ID in the argument `-g` to avoid errors.

- **Example #1**: to run inference on devices 2 and 5 with a max of 25 GiB on each, and make device 5 the primary device, add: `--gpu-id 5 -g 2:25 5:25`. In this example, not adding `--gpu-id 5` will give you an error.
- **Example #2**: to run inference on devices 0 and 3 with a max of 10GiB on 0 and 40GiB on 3, with device 0 as the primary device, add: `-g 0:10 3:40`. In this example, `--gpu-id` is not required because device 0 is specified in `-g`.
- **Example #3**: to run inference only on device 1 with a max of 75 GiB, add: `--gpu-id 1 -g 1:75`


## Running on consumer hardware
If you have multiple GPUs, each <48 GB VRAM, [the steps mentioned in this section on running on multiple GPUs](#running-on-multiple-gpus) still apply, unless, any of these apply:
- Running on just 1x GPU with <48 GB VRAM,
- <48 GB VRAM combined across multiple GPUs
- Running into Out-Of-Memory (OOM) issues

In which case, add the flag `-r CPU_RAM` where CPU_RAM is the maximum amount of RAM you'd like to allocate to loading model. Note: This significantly reduces inference speeds. 

The model will load without specifying `-r`, however, it is not recommended because it will allocate all available RAM to the model. To limit how much RAM the model can use, add `-r`.

If the total VRAM + CPU_RAM < the size of the model in GiB, the rest of the model will be offloaded to a folder "offload" at the root of the directory. Note: This significantly reduces inference speeds.

- Example: `-g 0:12 -r 20` will first load up to 12 GiB of the model into the CUDA device 0, then load up to 20 GiB into RAM, and load the rest into the "offload" directory.

How it works: 
- https://github.com/huggingface/blog/blob/main/accelerate-large-models.md
- https://www.youtube.com/embed/MWCSGj9jEAo
