## Fine-tuning with DeeperSpeed
### Install dependencies

`mamba install -c conda-forge cudatoolkit-dev`

`export CUDA_HOME=$CONDA_PREFIX`

`pip install evaluate datasets peft transformers git+https://github.com/EleutherAI/DeeperSpeed.git`

`pip install 'transformers[sklearn]'`

#### Install bitsandbytes if loading in 8-bit
`pip install bitsandbytes`

### Start...

`cd training/lora`

## Examples
#### From HuggingFace dataset:
```
deepspeed --num_gpus=1 finetune.py \
--deepspeed example/config.json \
--model_name_or_path togethercomputer/RedPajama-INCITE-Base-3B-v1 \
--dataset_name imdb \
--do_train \
--do_eval \
--fp16 \
--overwrite_cache \
--evaluation_strategy="steps" \
--output_dir finetuned \
--num_train_epochs 1 \
--eval_steps 15 \
--gradient_accumulation_steps 1 \
--per_device_train_batch_size 4 \
--use_fast_tokenizer True \
--learning_rate 1e-5 \
--warmup_steps 10
```
#### From train and validation files:
```
deepspeed --num_gpus=1 finetune.py \
--deepspeed example/config.json \
--model_name_or_path togethercomputer/RedPajama-INCITE-Base-3B-v1 \
--train_file train.csv \
--validation_file validation.csv \
--do_train \
--do_eval \
--fp16 \
--overwrite_cache \
--evaluation_strategy="steps" \
--output_dir finetuned \
--num_train_epochs 1 \
--eval_steps 15 \
--gradient_accumulation_steps 1 \
--per_device_train_batch_size 4 \
--use_fast_tokenizer True \
--learning_rate 1e-5 \
--warmup_steps 10
```

#### In 8-bit
** Change `fp16.enabled` to `false` in `example/config.json` **
```
deepspeed --num_gpus=1 finetune.py \
--deepspeed example/config.json \
--model_name_or_path togethercomputer/RedPajama-INCITE-Base-3B-v1 \
--dataset_name imdb \
--do_train \
--do_eval \
--int8 \
--low_cpu_mem_usage \
--overwrite_cache \
--evaluation_strategy="steps" \
--output_dir finetuned \
--num_train_epochs 1 \
--eval_steps 15 \
--gradient_accumulation_steps 1 \
--per_device_train_batch_size 4 \
--use_fast_tokenizer True \
--learning_rate 1e-5 \
--warmup_steps 10 \
--no_cache
```
