# Retrospex
The official code for Retrospex

## 0. Install dependencies

```bash
pip install -r requirements.txt
```
## 1. Train LLMs

### 1.1 Train Llama3(Webshop and ALFWorld)

Use ```alfworld/IL/train_llama3/run_lora_deepspeed.sh``` to train the model. You should change the model and dataset path in the script to your own path. And after the training, you can merge the lora part to the base model. Use ```alfworld/IL/train_llama3/merge.py``` to merge the model.

The link of the model we trained is : https://huggingface.co/AronXiang/RetrospexLLaMA3. This model is a merged one, you can directly call it by huggingface.

### 1.2 Train flan-t5(ScienceWorld)
Run ```ScienceWorld/IL/fast_agent/ds_train.sh``` to train the flan t5 large model.

The link of the model we trained is : https://drive.google.com/file/d/1U4NIxW9SalseBvKvNVMJe0jeqZutKSfb/view?usp=sharing

## 2. Train IQLs
See README.md in different environments.

## 3. Test Retrospex

### 3.1 ALFWorld

First, Install the ALFWorld Environment according to https://github.com/alfworld/alfworld.

Then you need to download and put datasets of ALFWorld into ```alfworld/alfworld_data```.

Then you can run ```alfworld/dynamic_action_scoring_alfworld.py``` to test the model.

### 3.2 ScienceWorld
First, Install the ScienceWorld Environment according to https://github.com/allenai/ScienceWorld.

Then you can run 
```bash
bash ScienceWorld/run_eval.sh
```
to test the model on all 30 subtasks.
Our code is refer to the code of original SWIFTSAGE: https://github.com/SwiftSage/SwiftSage, and we only occupy the Fast part——SWIFT with IQL added.

### 3.3 Webshop
First, you need to install the Webshop Environment according to https://github.com/princeton-nlp/WebShop.

Then you can run ```webshop/dynamic_action_scoring_alfworld.py``` to test the model.



