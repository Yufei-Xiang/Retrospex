First, Install the ScienceWorld Environment according to https://github.com/allenai/ScienceWorld.

## 1. Train flan-t5

Run ```ScienceWorld/IL/fast_agent/ds_train.sh``` to train the flan t5 large model.

## 2. Train IQL

Unzip the trajectories in memory_trajectories folder.
Use ```retrospect_scienceworld.py``` to train iql model.
Also, you can use the iql model in IQLmodel folder we used in our paper.

## 3. Test Retrospex

After training IQL models, then you can run 
```bash
bash run_eval.sh
```
to test the model on all 30 subtasks.

Our code is refer to the code of original SWIFTSAGE: https://github.com/SwiftSage/SwiftSage, and we only occupy the Fast part——SWIFT with IQL added.
