for seed in 0.01 0.005 0.0001
do
python main.py --iteration 100 --training_cycle 2000 --lr 0.0001 --dataset_dir './dataparser' --model_save_dir './trained_models' --c_r_class_num 3 --c_n_grid_size 10 --c_n_grid_channel 3 --coreset_buffer_size 200 --training_batch_size 8 --network_ensemble_cycle 5 --dropout_rate 0.25 --regularization_type 'ucl' --ucl_weight 0.1  --coreset_uncertainty_threshold $seed
done
