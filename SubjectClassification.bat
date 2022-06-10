python -u run.py --is_training 1 --model_id Subject_Classification --model Transformer --data PhysioNetSC --learning_rate 0.00001 --lr_adjust None --seq_len 640 --d_model 64 --root_path G:\subject_classification --label_path G:\subject_classification --c_out 105 --itr 1

python -u run.py --is_training 1 --model_id Subject_Classification --model Transformer --data PhysioNetSC --learning_rate 0.00001 --lr_adjust type1 --seq_len 640 --d_model 64 --root_path G:\subject_classification --label_path G:\subject_classification --c_out 105 --itr 1

python -u run.py --is_training 1 --model_id Subject_Classification --model Transformer --data PhysioNetSC --learning_rate 0.00001 --lr_adjust type2 --seq_len 640 --d_model 64 --root_path G:\subject_classification --label_path G:\subject_classification --c_out 105 --itr 1

python -u run.py --is_training 1 --model_id Subject_Classification --model FNet --data PhysioNetSC --learning_rate 0.00001 --lr_adjust None --seq_len 640 --d_model 64 --root_path G:\subject_classification --label_path G:\subject_classification --c_out 105 --itr 1

python -u run.py --is_training 1 --model_id Subject_Classification --model FNet --data PhysioNetSC --learning_rate 0.00001 --lr_adjust type1 --seq_len 640 --d_model 64 --root_path G:\subject_classification --label_path G:\subject_classification --c_out 105 --itr 1

python -u run.py --is_training 1 --model_id Subject_Classification --model FNet --data PhysioNetSC --learning_rate 0.00001 --lr_adjust type2 --seq_len 640 --d_model 64 --root_path G:\subject_classification --label_path G:\subject_classification --c_out 105 --itr 1

python -u run.py --is_training 1 --model_id Subject_Classification --model Autoformer --data PhysioNetSC --learning_rate 0.00001 --lr_adjust None --seq_len 640 --d_model 64 --root_path G:\subject_classification --label_path G:\subject_classification --c_out 105 --itr 1

python -u run.py --is_training 1 --model_id Subject_Classification --model Autoformer --data PhysioNetSC --learning_rate 0.00001 --lr_adjust type1 --seq_len 640 --d_model 64 --root_path G:\subject_classification --label_path G:\subject_classification --c_out 105 --itr 1

python -u run.py --is_training 1 --model_id Subject_Classification --model Autoformer --data PhysioNetSC --learning_rate 0.00001 --lr_adjust type2 --seq_len 640 --d_model 64 --root_path G:\subject_classification --label_path G:\subject_classification --c_out 105 --itr 1