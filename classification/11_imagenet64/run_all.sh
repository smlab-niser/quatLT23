#!/bin/bash

rm saved_models/RN18_real/* || echo no saved models in RN18 real to delete
tmux new-session -d -s R18 "python R18.py"
echo "R18.py ran on tmux session R18 - waiting 2s..."
sleep 2s  # wait so that they go to wandb in order

rm saved_models/RN18_quat/* || echo no saved models in RN18 quat to delete
tmux new-session -d -s Q18 "python Q18.py"
echo "Q18.py ran on tmux session Q18 - waiting 2s..."
sleep 2s  # wait so that they go to wandb in order

rm saved_models/RN152_real/* || echo no saved models in RN152 real to delete
tmux new-session -d -s R152 "python R152.py"
echo "R152.py ran on tmux session R152 - waiting 2s..."
sleep 2s  # wait so that they go to wandb in order

rm saved_models/RN152_quat/* || echo no saved models in RN152 quat to delete
tmux new-session -d -s Q152 "python Q152.py"
echo "Q152.py ran on tmux session Q152 - waiting 2s..."