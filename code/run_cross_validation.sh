
python3 trainVal.py -c cross_validation_config.ini --exp_id cross_validation --model_id resnet_split$1      --temp_mod linear --feat resnet18    --split $1
python3 trainVal.py -c cross_validation_config.ini --exp_id cross_validation --model_id resnet-lstm_split$1 --temp_mod lstm   --feat resnet18    --split $1
python3 trainVal.py -c cross_validation_config.ini --exp_id cross_validation --model_id resnet3D_split$1    --temp_mod linear --feat r2plus1d_18 --split $1

#python3 processResults.py --exp_id cross_validation --model_ids resnet_split$1 resnet-lstm_split$1 resnet3D_split$1 