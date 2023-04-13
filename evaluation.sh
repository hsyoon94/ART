for dataset in 'cifar10' 'cifar100' 'mnist'
do
    python main.py --regularization_type 'l2' --coreset_buffer_size 100 --coreset_type 'fifo' --experiment $dataset &
    python main.py --regularization_type 'l2' --coreset_buffer_size 100 --coreset_type 'ucm' --experiment $dataset &
    python main.py --regularization_type 'l2' --coreset_buffer_size 100 --coreset_type 'reservoir' --experiment $dataset &
    python main.py --regularization_type 'ucl' --coreset_buffer_size 100 --coreset_type 'fifo' --experiment $dataset &
    python main.py --regularization_type 'ucl' --coreset_buffer_size 100 --coreset_type 'ucm' --experiment $dataset &
    python main.py --regularization_type 'ucl' --coreset_buffer_size 100 --coreset_type 'reservoir' --experiment $dataset
done