# echo -n "rm -rf " > saved_models/dirs.sh

# tmux new-session -d -s yolo01 "python pretraining.py --gpu 0 --lr 0.01 --momentum 0.90 --weight_decay 0.001 --optimiser  sgd";echo yolo01;sleep 1
# tmux new-session -d -s yolo03 "python pretraining.py --gpu 1 --lr 0.03 --momentum 0.90 --weight_decay 0.001 --optimiser  sgd";echo yolo03;sleep 1
# tmux new-session -d -s yolo02 "python pretraining.py --gpu 0 --lr 0.01 --momentum 0.90 --weight_decay 0.01  --optimiser  sgd";echo yolo02;sleep 1
# tmux new-session -d -s yolo04 "python pretraining.py --gpu 1 --lr 0.03 --momentum 0.90 --weight_decay 0.01  --optimiser  sgd";echo yolo04;sleep 1

# tmux new-session -d -s yolo1 "python pretraining.py --gpu 0 --lr 0.01 --momentum 0.95 --weight_decay 0.001 --optimiser  sgd";echo yolo1;sleep 1
# tmux new-session -d -s yolo2 "python pretraining.py --gpu 0 --lr 0.03 --momentum 0.95 --weight_decay 0.001 --optimiser  sgd";echo yolo2;sleep 1
tmux new-session -d -s yolo3 "python pretraining.py --gpu 1 --lr 0.05 --momentum 0.90 --weight_decay 0.001 --optimiser  sgd";echo yolo3;sleep 1
tmux new-session -d -s yolo4 "python pretraining.py --gpu 1 --lr 0.1  --momentum 0.90 --weight_decay 0.001 --optimiser  sgd";echo yolo4;sleep 1

# tmux new-session -d -s yolo5 "python pretraining.py --gpu 2 --lr 0.01 --momentum 0.95 --weight_decay 0.0001 --optimiser sgd";echo yolo5;sleep 1
# tmux new-session -d -s yolo6 "python pretraining.py --gpu 2 --lr 0.03 --momentum 0.95 --weight_decay 0.0001 --optimiser sgd";echo yolo6;sleep 1
tmux new-session -d -s yolo7 "python pretraining.py --gpu 3 --lr 0.05 --momentum 0.90 --weight_decay 0.0001 --optimiser sgd";echo yolo7;sleep 1
tmux new-session -d -s yolo8 "python pretraining.py --gpu 3 --lr 0.1  --momentum 0.90 --weight_decay 0.0001 --optimiser sgd";echo yolo8;sleep 1


# echo -n "All jobs started, waiting for 10 seconds before making saving directories... "
# sleep 10
# cd saved_models
# cat dirs > rm -rf
# echo deleted
# sleep 10 
# cat dirs > mkdir
# cd ..
# echo "Done."
