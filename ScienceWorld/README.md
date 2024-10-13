First, Install the ScienceWorld Environment according to https://github.com/allenai/ScienceWorld.

## 1. Train flan-t5

Run ```ScienceWorld/IL/fast_agent/ds_train.sh``` to train the flan t5 large model.

## 2. Test Retrospex

After training IQL models, then you can run 
```bash
bash swift_inference/run_eval_fast_slow.sh
```
to test the model on all 30 subtasks.

Our code is refer to the code of original SWIFTSAGE: https://github.com/SwiftSage/SwiftSage, and we only occupy the Fast part——SWIFT with IQL added.