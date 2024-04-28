"""
Filename: Llama2-70B-Chat-finetune-LoRA.py
Author: Szymon Manduk
Company: Szymon Manduk AI, manduk.ai
Description: 
    - Fine-tuning 'meta-llama/Llama-2-70b-chat-hf' using PEFT (LoRA and 4-bit quantization)
    - Training done on the proprietary English conversational dataset
    - 400 steps x 4 batch size = 1600 data samples, 1600 / 9347 = 17% of the dataset
    - uses AutoTokenizer -> LlamaTokenizerFast. 
    - use bnb_4bit_compute_dtype=torch.bfloat16 to speed up training
    - use gradient_accumulation_steps=1 to speed up training (with value = 4 performance was terrible)
    - we PAD to multiple of 16
    - In order to execute script:
        1. Upload script to a server (requires GPU with 80GB of VRAM)
        2. Upload .env (same dir)
        3. Upload dataset (same dir)
        4. Create 'Models' directory
        5. Activate the environment
        6. pip install -q transformers datasets accelerate peft bitsandbytes python-dotenv sentencepiece wandb scipy protobuf
        7. Run in the background: nohup python Llama2-70B-Chat-finetune-LoRA.py > output.log 2>&1 &
        8. To view the output: tail -f output.log
License: This project utilizes a dual licensing model: GNU GPL v3.0 and Commercial License. For detailed information on the license used, refer to the LICENSE, RESOURCE-LICENSES and README.md files.
Copyright (c) 2024 Szymon Manduk AI.
"""

print('Script started.')

import torch
import os
import time
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import wandb
print('Imports done.')

#########################################
############# CONFIGURATION #############

BASE_MODEL_NAME='meta-llama/Llama-2-70b-chat-hf'
MY_MODEL_NAME='Llama-2-70b-assistant-medical-facility-EN-C'
COMMIT_MESSAGE="Llama2 Chat 70B finetuned on proprietary dataset v.C"  # commit message for the Huggingface Hub
WANDB_PROJECT_NAME='llama2-70B-peft-english'
WANDB_RUN_NAME='09122023-C'
SAVE_MODEL_LOCALLY=True                                                   
SAVE_MODEL_TO_HUB=True                                                    
ENV_FILE='./.env'                                                         
HF_TOKEN_NAME='HF_SECRET'   
WANDB_TOKEN_NAME='WANDB_SECRET'  
VERBOSE=True  # if True, print more stuff
SLEEP_TIME=5  # time to sleep between steps, so we can read the output
JSON_FILE_PATH = "English-dataset-llama2-instruction-format-2-train.json"   # data
OUTPUT_DIR='./Models/'  
# Previous analysis of dataset shows that the maxiumu length for instruction format is 795 tokens. 
# But we need also give some room for future RAG tokens, so we aim at 1280 tokens.
MAX_LENGTH=1280          # used by a tokenizer - max number of tokens in a sequence (should be a multiple of 16)
PAD_TO_MULTIPLE_OF=16    # used by a tokenizer - pad input to multiple of this number of tokens
# If we want to continue training from a checkpoint, we need to set RESUME_FROM_CHECKPOINT to True and provide the name of the model to load
RESUME_FROM_CHECKPOINT=False
FINETUNED_MODEL_NAME=''  # if we resume from checkpoint, we need to provide the name of the model to load


# llama2 is a gated model, we need auth token. 
_ = load_dotenv(find_dotenv(filename=ENV_FILE))
hf_api_key  = os.environ[HF_TOKEN_NAME]
wand_api_key  = os.environ[WANDB_TOKEN_NAME]
if VERBOSE:
    print(f'API keys loaded')

wandb.login(key=wand_api_key)
wandb.init(project=WANDB_PROJECT_NAME, name=WANDB_RUN_NAME)

# Helper function to print number of different parameters in the model
def count_model_params(model):
    fp32_params = 0
    int8_params = 0
    bfloat16_params = 0
    float16_params = 0
    other_params = 0
    requires_grad_params = 0

    for param in model.parameters():
        if param.dtype == torch.float32:
            fp32_params += param.numel()
        if param.dtype == torch.int8:
            int8_params += param.numel()
        if param.dtype == torch.bfloat16:
            bfloat16_params += param.numel()
        if param.dtype == torch.float16:
            float16_params += param.numel()
        if param.dtype != torch.float32 and param.dtype != torch.int8 and param.dtype != torch.bfloat16 and param.dtype != torch.float16:
            other_params += param.numel()
        if param.requires_grad:
            requires_grad_params += param.numel()

    return fp32_params, int8_params, bfloat16_params, float16_params, other_params, requires_grad_params

##########################################################
############# MODEL, QUANTIZATION, TOKENIZER #############

model_config = AutoConfig.from_pretrained(
    BASE_MODEL_NAME,
    token=hf_api_key,
)
if VERBOSE:
    print(f'Model config loaded')
    print(model_config)
    time.sleep(SLEEP_TIME)

# We plan to train llama2 70B on 1xA100 80GB, so we need to use mixed precision and 4-bit quantization
quantization_config = BitsAndBytesConfig(
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    load_in_4bit=True,
)
if VERBOSE:
    print(f'Quantization config loaded')
    print(quantization_config)
    time.sleep(SLEEP_TIME)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    config=model_config,
    load_in_4bit=True,
    quantization_config=quantization_config,
    token=hf_api_key,
    device_map='auto'
)
if VERBOSE:
    print(f'Model loaded.')
    print(model.config)
    time.sleep(SLEEP_TIME*2)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_NAME,
    add_eos_token = True, # we need the tokenizer to add eos token at the end of the prompt
    token=hf_api_key,
)
if VERBOSE:
    print(f'Tokenizer loaded.')
    print(tokenizer)
    time.sleep(SLEEP_TIME)

# Setting PAD token
if VERBOSE:
    print('Padding before:')
    print(tokenizer.pad_token)
    print(tokenizer.pad_token_id)
    print(tokenizer.padding_side)

# we need to extend the tokenizer with [pad] token ...
tokenizer.add_special_tokens(
    {
        "pad_token": "[PAD]",
    }
)
tokenizer.padding_side = "left"

# Extend the model vocab. Parameter pad_to_multiple_of is needed to make the model work efficiently on GPU
# For FP16 it's 64 https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=PAD_TO_MULTIPLE_OF)  

if VERBOSE:
    print('Padding after:')
    print(tokenizer.pad_token)
    print(tokenizer.pad_token_id)
    print(tokenizer.padding_side)
    time.sleep(SLEEP_TIME)


# Short summary of the model's parameters
if VERBOSE:
    fp32_params, int8_params, bfloat16_param, float16_param, other_params, requires_grad_params = count_model_params(model)
    print(f"\n\nFP32 parameters: {fp32_params}")
    print(f"INT8 parameters: {int8_params}")
    print(f"BFLOAT16 parameters: {bfloat16_param}")
    print(f"FLOAT16 parameters: {float16_param}")
    print(f"Other parameters: {other_params}")
    print(f"Parameters requiring gradients: {requires_grad_params}")
    time.sleep(SLEEP_TIME*2)

if VERBOSE:
    print(f'Watch out model.config.pad_token_id before:{model.config.pad_token_id}. SHOULD WE SET IT?')
model.config.pad_token_id = tokenizer.pad_token_id
if VERBOSE:
    print(f'model.config.pad_token_id after:{model.config.pad_token_id}')
    time.sleep(SLEEP_TIME)


##########################################################
####################### PEFT (LoRA) ######################

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    # previous experiments show that the best results come from using lora on attention heads and model head - this might require more experimentation
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "lm_head"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
if VERBOSE:
    print(f"LoRA config loaded.")
    print(lora_config)
    time.sleep(SLEEP_TIME)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': True})
if VERBOSE:
    print("Model prepared for kbit training.")
    print("Model params after kbit training.")
    fp32_params, int8_params, bfloat16_param, float16_param, other_params, requires_grad_params = count_model_params(model)
    print(f"\n\nFP32 parameters: {fp32_params}")
    print(f"INT8 parameters: {int8_params}")
    print(f"BFLOAT16 parameters: {bfloat16_param}")
    print(f"FLOAT16 parameters: {float16_param}")
    print(f"Other parameters: {other_params}")
    print(f"Parameters requiring gradients: {requires_grad_params}")
    time.sleep(SLEEP_TIME)

if RESUME_FROM_CHECKPOINT:
    model = PeftModel.from_pretrained(
        model, 
        FINETUNED_MODEL_NAME,
        token=hf_api_key,
    )
    print(f"PeftModel loaded from pretrained: {FINETUNED_MODEL_NAME}.")
    print(model.config)
    print(model)
else:
    # new model
    model = get_peft_model(model, lora_config)
    if VERBOSE:
        print("NEW LoRA adaptor added!")
        print(model)
        model.print_trainable_parameters()
        time.sleep(SLEEP_TIME*2)

# Short summary of the model's parameters
if VERBOSE:
    fp32_params, int8_params, bfloat16_param, float16_param, other_params, requires_grad_params = count_model_params(model)
    print(f"\n\nFP32 parameters: {fp32_params}")
    print(f"INT8 parameters: {int8_params}")
    print(f"BFLOAT16 parameters: {bfloat16_param}")
    print(f"FLOAT16 parameters: {float16_param}")
    print(f"Other parameters: {other_params}")
    print(f"Parameters requiring gradients: {requires_grad_params}")
    time.sleep(SLEEP_TIME*2)

##################################################
######## DATASET, PROMPT, TOKENIZATION ###########

# Using load_dataset from the datasets library to load the dataset from the json file, instead of just using json.load, because
# datasets library has a lot of useful features, like splitting the dataset into train, validation and test sets, shuffling and map function
dataset = load_dataset('json', data_files=JSON_FILE_PATH, )
if VERBOSE:
    print(f"Dataset loaded. Contains {len(dataset['train'])} samples.")
    time.sleep(SLEEP_TIME)

# shuffle train dataset - we don't use seed to shuffle differently every time
# this seems unnecessary, because we later use train_test_split, which shuffles the dataset anyway
dataset = dataset.shuffle()

prompts = []
# PROMPT TEMPLATE 1
prompt1 = (
f"""You are a helpful Assistant on the hotline of a medical facility. Your task is to talk to a Patient. Below is the history of the conversation with a Patient.

History of the conversation (below between four hashes):
#### 
{{conversation}}
####

The Patient asks another question (below between four hashes):
####
{{question}}
####

Keep your answer short and precise. 
Do not continue as a Patient, just respond as an Assistant. 
Answer the Patient's question as an Assistant. 

Response: {{response}}"""
)
prompts.append(prompt1)

# PROMPT TEMPLATE 2
prompt2 = (
f"""You are a helpful Assistant on the hotline of a medical facility. Your task is to talk to a Patient. Below is the history of the conversation with a Patient (between four hashes): #### {{conversation}} ####
The Patient asks now another question (below between four hashes): #### {{question}} ####
Respond as an Assistant. Keep your answer short and precise. Do not continue as a Patient, just respond as an Assistant. Answer the Patient's question as an Assistant. 
Response: {{response}}"""
)
prompts.append(prompt2)
prompts.append(prompt2)  # twice to increase the probability of this prompt

# PROMPT TEMPLATE 3
prompt3 = (
f"""You are a helpful Assistant on the hotline of a medical facility. Your task is to talk to a Patient. Here are additional information you may need between four asterisks: **** ****
Below is the history of the conversation with a Patient (between four hashes): #### {{conversation}} ####
The Patient asks now another question (below between four hashes): #### {{question}} ####
Respond as an Assistant. Keep your answer short and precise. Do not continue as a Patient, just answer the Patient's question as an Assistant. 
Response: {{response}}"""
)
prompts.append(prompt3)
prompts.append(prompt3)  # twice to increase the probability of this prompt

# Helper function to apply the prompt template to a sample
def apply_prompt_template(sample): 
    # randomly choose one of the prompts
    prompt = prompts[torch.randint(0, len(prompts), (1,)).item()]

    label = sample['label']
    if label.startswith("Assistant: "):
        label = label[len("Assistant: "):]  # Remove the "Assistant: " prefix

    return {
        "query": prompt.format(
            conversation=sample['conversation_history'],
            question=sample['input'],
            response=label
        ),
    }


# Apply prompt template to the dataset and split
dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset["train"].features.keys()))
split_dataset = dataset['train'].train_test_split(test_size=0.025)  # 2.5% of the train dataset will be used for validation
train_dataset, val_dataset = split_dataset['train'], split_dataset['test']
if VERBOSE:
    print(len(train_dataset), len(val_dataset))
    time.sleep(SLEEP_TIME)

# The last step is to tokenize the dataset. We need to apply padding and truncation. 
train_dataset = train_dataset.map(
    lambda sample: {
        **tokenizer(sample["query"], padding="max_length", truncation=True, max_length=MAX_LENGTH, pad_to_multiple_of=PAD_TO_MULTIPLE_OF),
    },
    batched=True,
    remove_columns=list(train_dataset.features),
)
if VERBOSE:
    print(f'The length of the train dataset is {len(train_dataset)}. Maximum lenght of a sample: {max([len(x) for x in train_dataset["input_ids"]])}')
    time.sleep(SLEEP_TIME*2)

# The same step for the validation dataset
val_dataset = val_dataset.map(
    lambda sample: {
        **tokenizer(sample["query"], padding="max_length", truncation=True, max_length=MAX_LENGTH, pad_to_multiple_of=PAD_TO_MULTIPLE_OF),
    },
    batched=True,
    remove_columns=list(val_dataset.features),
)
if VERBOSE:
    print(f'The length of the validation dataset is {len(val_dataset)}. Maximum lenght of a sample: {max([len(x) for x in val_dataset["input_ids"]])}')
    time.sleep(SLEEP_TIME)


########################################################
####################### TRAINING #######################

# Training Parameters
# For batch_size=4 they use 63GB of VRAM on A100. 
# 2 steps take around 600s for gradient_accumulation_steps=4, so 150s for gradient_accumulation_steps=1
# 400 steps x 4 batch size = 1600 data samples, 1600 / 9347 = 17% of the dataset
# 400 steps x 150s = 60000s = 1000min = ~16h = ~$35 on RunPod A100 80GB
training_args = TrainingArguments(
    learning_rate=3e-4,             # 0.0003
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=4, 
    # gradient_accumulation_steps=4,  # not used, because it slows down training
    max_steps=400,     
    warmup_steps=50,               
    logging_steps=2,               
    evaluation_strategy="steps",    
    eval_steps=2500,                 # I don't want to evaluate, so I set it to a big number
    output_dir=OUTPUT_DIR + MY_MODEL_NAME,
    report_to="wandb",
    save_steps=135,                 # training is expensive, so we save 1/3 and 2/3 of the training
)

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# turn off caching to save RAM
model.config.use_cache = False 

trainer.train()

if SAVE_MODEL_LOCALLY:
    model.save_pretrained(f"{OUTPUT_DIR}{MY_MODEL_NAME}")

wandb.finish()

if SAVE_MODEL_TO_HUB:
    model.push_to_hub(MY_MODEL_NAME, use_auth_token=hf_api_key, commit_message=COMMIT_MESSAGE)

print("Script finished.")