python -u run.py --is_training 1 --model_id CV --model GatedTransformer --data PhysioNetCV640 --learning_rate 0.001 --lr_adjust type1 --seq_len 640 --d_model 64 --label_len 64 --d_spatial_model 640

python -u run.py --is_training 1 --model_id CV --model GatedTransformer --data PhysioNetCV640 --learning_rate 0.001 --lr_adjust type2 --seq_len 640 --d_model 64 --label_len 64 --d_spatial_model 640

python -u run.py --is_training 1 --model_id CV --model GatedTransformer --data PhysioNetCV640 --learning_rate 0.0001 --lr_adjust type1 --seq_len 640 --d_model 64 --label_len 64 --d_spatial_model 640

python -u run.py --is_training 1 --model_id CV --model GatedTransformer --data PhysioNetCV640 --learning_rate 0.0001 --lr_adjust type2 --seq_len 640 --d_model 64 --label_len 64 --d_spatial_model 640

python -u run.py --is_training 1 --model_id CV --model GatedTransformer --data PhysioNetCV32 --learning_rate 0.001 --lr_adjust type1 --root_path G:\data_swm

python -u run.py --is_training 1 --model_id CV --model GatedTransformer --data PhysioNetCV32 --learning_rate 0.001 --lr_adjust type2 --root_path G:\data_swm

python -u run.py --is_training 1 --model_id CV --model GatedTransformer --data PhysioNetCV32 --learning_rate 0.0001 --lr_adjust type1 --root_path G:\data_swm

python -u run.py --is_training 1 --model_id CV --model GatedTransformer --data PhysioNetCV32 --learning_rate 0.0001 --lr_adjust type2 --root_path G:\data_swm