import argparse

# python main.py --iteration 10000 --training_cycle 10 --coreset_buffer_size 10 --lr 0.0001 --dataset_dir './dataparser' --model_save_dir './trained_models' --c_r_class_num 1 --c_n_grid_size 10 --c_n_grid_channel 3

def get_args():
    parser = argparse.ArgumentParser(description="Arguments of UFO")

    parser.add_argument("--iteration", default=10000, type=int, help='total training iteration')
    parser.add_argument("--training_cycle", default=10, type=int, help='training_cycle')
    parser.add_argument("--coreset_buffer_size", default=10, type=int, help="batch size")
    parser.add_argument("--c_r_class_num", default=1, type=int, help="batch size")
    parser.add_argument("--c_n_grid_size", default=10, type=int, help="batch size")
    parser.add_argument("--c_n_grid_channel", default=3, type=int, help="batch size")
    parser.add_argument("--lr", default=0.0001, type=float, help="batch size")
    parser.add_argument("--dataset_dir", default='./dataparser', help="dataset directory")
    parser.add_argument("--model_save_dir", default='./trained_models', help="trained model save directory")
    
    args = parser.parse_args()

    return args

