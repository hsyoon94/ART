import numpy as np
import csv

csv_file_name = '/home/hsyoon94/Downloads/wandb_export_2023-04-04T16_01_55.115+09_00.csv'
value_list = []
with open(csv_file_name, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        try:
            value_list.append(float(row[1]))
        except:
            pass
        
    np_value_list = np.array(value_list)
    print(np_value_list)
    value_mean = np.mean(np_value_list)
    value_std = np.std(np_value_list)

    print("value_mean", value_mean)
    print("value_std", value_std)