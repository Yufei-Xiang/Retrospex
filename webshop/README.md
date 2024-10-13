First, you need to install the Webshop Environment according to https://github.com/princeton-nlp/WebShop.

## 1. Train LLaMA3

The LLM for Webshop is the same as ALFWorld. So you can refer to ```alfworld/README.md``` to train the LLaMA3 model.

Use ```alfworld/IL/train_llama3/run_lora_deepspeed.sh``` to train the model. You should change the model and dataset path in the script to your own path. And after the training, you can merge the lora part to the base model. Use ```alfworld/IL/train_llama3/merge.py``` to merge the model.

## 2. Train IQL

Use ```retrospect_webshop.py``` to train iql model.

## 3. Test Retrospex

After training IQL models, run ```dynamic_action_scoring_webshop.py``` to test our method.