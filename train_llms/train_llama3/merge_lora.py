import os

import torch
import transformers
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL = None
# TODO(path of base model like llama3)
LORA_MODEL = None
# TODO(path of lora part)
OUTPUT_DIR = None
# TODO(output path of merged model)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL,trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cuda()

lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL,
    torch_dtype=torch.float16
)

model = lora_model.merge_and_unload()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
