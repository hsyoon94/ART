import numpy as np
import torch
from algo.SIRT import SIRT

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

def main():
    network = SIRT(5, 3, 3, device)
    print(network)

if __name__ == '__main__':
    main()