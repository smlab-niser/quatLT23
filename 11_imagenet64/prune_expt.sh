#!/bin/bash

# # for RN18 in real
# {rm saved_models/RN18_real_prune/* && echo cleaned previously saved models;} || {echo no saved models to delete;}
# tmux new-session -d -s R18 "python img64.py RN18_real --save --log --gpu 0"
# echo RN18 real ran for all sparsities on tmux session R18
# sleep 2s  # wait so that they go to wandb in order


# # for RN18 in quat
# {
# 	rm saved_models/RN18_quat_prune/* &&
# 	echo cleaned previously saved models
# } || 
# {
# 	echo no saved models to delete
# }
# tmux new-session -d -s Q18 "python img64.py RN18_quat --save --log --gpu 1"
# echo RN18 quat ran for all sparsities on tmux session Q18
# # sleep 2s  # wait so that they go to wandb in order


# # for RN152 in real
# {
# 	rm saved_models/RN152_real_prune/* &&
# 	echo cleaned previously saved models
# } || 
# {
# 	echo no saved models to delete
# }
# tmux new-session -d -s R152 "python img64.py RN152_real --save --log --gpu 2"
# echo RN152 real ran for all sparsities on tmux session R152
# sleep 2s


# # for RN152 in quat
# {
# 	rm saved_models/RN152_quat_prune/* &&
# 	echo cleaned previously saved models
# } || 
# {
# 	echo no saved models to delete
# }
# tmux new-session -d -s Q152 "python img64.py RN152_quat --save --log --gpu 3"
# echo RN152 quat ran for all sparsities on tmux session Q152


# tmux new-session -d -s R34 "python img64.py RN34_real --save --log --gpu 0"
# tmux new-session -d -s Q34 "python img64.py RN34_quat --save --log --gpu 1"
# tmux new-session -d -s R152 "python img64.py RN152_real --save --log --gpu 2"
# tmux new-session -d -s Q152 "python img64.py RN152_quat --save --log --gpu 3"

tmux new-session -d -s Q50 "python img64.py RN50_quat --save --log --gpu 0"
tmux new-session -d -s Q50 "python img64.py RN50_quat --save --log --gpu 0"