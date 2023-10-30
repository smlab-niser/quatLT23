tmux new-session -d -s grids1 "python map_th_gridsearch.py -m real -e 120 --gpu 0"
echo grids1
tmux new-session -d -s grids2 "python map_th_gridsearch.py -m real -e 140 --gpu 1"
echo grids2
tmux new-session -d -s grids3 "python map_th_gridsearch.py -m quat -e 120 --gpu 2"
echo grids3
tmux new-session -d -s grids4 "python map_th_gridsearch.py -m quat -e 140 --gpu 3"
echo grids4