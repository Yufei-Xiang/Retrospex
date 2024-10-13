First, Install the ALFWorld Environment according to https://github.com/alfworld/alfworld.

And you need to download and put datasets of ALFWorld data into ```alfworld_data/``` in this folder.

## 1. Train LLaMA3

Use ```IL/train_llama3/run_lora_deepspeed.sh``` to train the model. You should change the model and dataset path in the script to your own path. And after the training, you can merge the lora part to the base model. Use ```IL/train_llama3/merge.py``` to merge the model.

## 2. Test Retrospex

After training IQL models, run ```alfworld/alfworldtest_addiql.py``` to test our method.