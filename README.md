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
Run ```swift_only/fast_agent/ds_train.sh``` to train the flan t5 large model.

## 2. Train IQLs
Run ```train_IQL/train_iql_swift.py``` to train the IQL model for ScienceWorld, pay attention that you need to first unzip the collected trajectories of ScienceWorld. For Webshop and ALFWorld, directly run the corresponding python file in train_IQL.

## 3. Test Retrospex

### 3.1 ALFWorld

First you need to download and put datasets of ALFWorld into ```alfworld/alfworld_data```.

Then you can run ```alfworld/alfworldtest_addiql.py``` to test the model.

### 3.2 ScienceWorld
First, Install the ScienceWorld Environment according to https://github.com/allenai/ScienceWorld.
Then you can run 
```bash
bash swift_inference/run_eval_fast_slow.sh
```
to test the model on all 30 subtasks.
Our code is refer to the code of original SWIFTSAGE: https://github.com/SwiftSage/SwiftSage, and we only occupy the Fast part——SWIFT with IQL added.

### 3.3 Webshop
 First, you need to install the Webshop Environment according to https://github.com/princeton-nlp/WebShop.
You can run ```webshop/webshoptest_addiql.py``` to test the model.



