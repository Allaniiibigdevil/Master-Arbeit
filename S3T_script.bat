python -u run.py --is_training 1 --model_id CV --model S3T --data PhysioNetCV640 --learning_rate 0.00001 --lr_adjust None --seq_len 640 --d_model 64 --d_spatial_model 640

python -u run.py --is_training 1 --model_id CV --model S3T --data PhysioNetCV640 --learning_rate 0.00001 --lr_adjust type1 --seq_len 640 --d_model 64 --d_spatial_model 640

python -u run.py --is_training 1 --model_id CV --model S3T --data PhysioNetCV640 --learning_rate 0.00001 --lr_adjust type2 --seq_len 640 --d_model 64 --d_spatial_model 640

python -u run.py --is_training 1 --model_id CV --model S3T --data PhysioNetCV640 --learning_rate 0.000001 --lr_adjust None --seq_len 640 --d_model 64 --d_spatial_model 640

python -u run.py --is_training 1 --model_id CV --model S3T --data PhysioNetCV640 --learning_rate 0.000001 --lr_adjust type1 --seq_len 640 --d_model 64 --d_spatial_model 640

python -u run.py --is_training 1 --model_id CV --model S3T --data PhysioNetCV640 --learning_rate 0.000001 --lr_adjust type2 --seq_len 640 --d_model 64 --d_spatial_model 640