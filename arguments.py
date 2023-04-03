import argparse

# python main.py --iteration 100 --training_cycle 2000 --lr 0.0001 --dataset_dir './dataparser' --model_save_dir './trained_models' --c_r_class_num 3 --c_n_grid_size 10 --c_n_grid_channel 3 --coreset_buffer_size 200 --training_batch_size 8 --network_ensemble_cycle 5 --dropout_rate 0.25 --regularization_type 'ucl' --ucl_weight 0.2

def get_args():
    parser = argparse.ArgumentParser(description="Arguments of UFO")

    parser.add_argument("--iteration", default=10000, type=int, help='total training iteration')
    parser.add_argument("--training_cycle", default=10, type=int, help='training_cycle')
    parser.add_argument("--training_batch_size", default=8, type=int, help='training_cycle')
    parser.add_argument("--coreset_buffer_size", default=10, type=int, help="batch size")
    parser.add_argument("--coreset_type", default='fifo', help="coreset management type")
    parser.add_argument("--network_ensemble_cycle", default=5,  type=int, help="coreset management type")
    parser.add_argument("--dropout_rate", default=0.1,  type=float, help="coreset management type")
    parser.add_argument("--regularization_type", default='ucl', help="regularization type")
    parser.add_argument("--ucl_weight", default=0.1, type=float, help="regularization type")
    parser.add_argument("--c_r_class_num", default=1, type=int, help="batch size")
    parser.add_argument("--c_n_grid_size", default=10, type=int, help="batch size")
    parser.add_argument("--c_n_grid_channel", default=3, type=int, help="batch size")
    parser.add_argument("--lr", default=0.001, type=float, help="batch size")
    parser.add_argument("--dataset_dir", default='./dataparser', help="dataset directory")
    parser.add_argument("--model_save_dir", default='./trained_models', help="trained model save directory")

    args = parser.parse_args()

    return args