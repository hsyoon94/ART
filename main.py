import numpy as np
import torch
from algo.ART import ART
from algo.coreset 
import csv

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

def main():
    # Network definition
    network = ART(5, 3, 3, device)    

    # Coreset Initialization with minimum # of data
    f = open("dataparser/data.csv")
    rdr = csv.reader(f)

    for line in rdr:
        tmp_ipt = line[0]
        tmp_opt = line[1]        
        
    # Training starts with coreset

        # Training with current coreset
        
        # update coreset

if __name__ == '__main__':
    main()