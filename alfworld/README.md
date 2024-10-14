First, Install the ALFWorld Environment according to https://github.com/alfworld/alfworld.

And you need to download and put datasets of ALFWorld data into ```alfworld_data/``` in this folder.

## 1. Train LLaMA3

Before training, you need to download AgentInstruct and ShareGPT datasets and reprocess it.

Then use ```IL/train_llama3/run_lora_deepspeed.sh``` to train the model. You should change the model and dataset path in the script to your own path. And after the training, you can merge the lora part to the base model. Use ```IL/train_llama3/merge.py``` to merge the model.
The link of the model we trained is : https://huggingface.co/AronXiang/RetrospexLLaMA3

## 2. Train IQL

Use ```retrospect_alfworld.py``` to train iql model.
Also, you can use the iql model in IQLmodel folder we used in our paper.

## 3. Test Retrospex

After training IQL models, run ```dynamic_action_scoring_alfworld.py``` to test our method.
