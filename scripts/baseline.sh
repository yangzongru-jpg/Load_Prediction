export CUDA_VISIBLE_DEVICES=2

#GD的数据
python -u /home/yzr/msdis/Load_Prediction/run.py --dataset 'GD' --epochs 10
# SD的数据
python -u /home/yzr/msdis/Load_Prediction/run.py --dataset 'SD' --epochs 10





