# Retrospex
The official code for Retrospex

## 0. Install dependencies

```bash
pip install -r requirements.txt
```
You need to carefully install ALFWorld and Webshop environment.

## 1. Train LLMs

### 1.1 Train Llama3(Webshop and ALFWorld)

Use ```train_llama3/run_lora_deepspeed.sh``` to train the model. You should change the model and dataset path in the script to your own path. And after the training, you should merge the lora part to the base model. Use ```train_llama3/merge.py``` to merge the model.

### 1.2 Train flan-t5(ScienceWorld)

## 2. Train IQLs

## 3. Test Retrospex

### 3.1 ALFWorld

First you need to download and put datasets of ALFWorld into ```alfworld/alfworld_data```.

Then you can run ```alfworld/alfworldtest_addiql.py``` to test the model.

### 3.2 ScienceWorld

### 3.3 Webshop

You can run ```webshop/webshoptest_addiql.py``` to test the model.



