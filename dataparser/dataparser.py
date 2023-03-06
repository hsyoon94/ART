#!/usr/bin/env python3

import numpy as np
import rosbag
from std_msgs.msg import Int32, String
from matplotlib import pyplot as plt
from scipy import signal
import math
import csv
from algo.coreset import CoresetBuffer
import os

bag = rosbag.Bag('/media/hsyoon94/HS_SDD/bagfiles/230201/visky_2023-02-01-15-34-49.bag')

odom_topic = '/aft_mapped_to_init'
costmap_topic = '/traversability_costmap_visualize'
imu_topic = '/imu/data'

costmap_data = list()
odom_data = list()
imu_data = list()

def main():
    
    file_write = open('data_c_r.csv', 'w', newline="")
    file_writer = csv.writer(file_write)

    occupancy_grid = np.array([])
    
    for topic, msg, t in bag.read_messages(topics=[costmap_topic, odom_topic, imu_topic]):
        if topic == costmap_topic:
            costmap_data.append(msg.data)
        
        elif topic == odom_topic:
            odom_data.append(msg)

        elif topic == imu_topic:
            # imu_tmp = list([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
            imu_data.append(msg.orientation.z)

    len_costmap = len(costmap_data)
    len_imu = len(imu_data)
    len_odom = len(odom_data)

    count = 0

    for costmap_data_element in costmap_data:

        input_1d_array_len = len(costmap_data_element)
        
        output_2d_array_size = math.sqrt(input_1d_array_len)
        output_2d_array_size = int(output_2d_array_size)
        output_2d_array = np.zeros((output_2d_array_size, output_2d_array_size))

        for row_idx in range(output_2d_array_size):
            for col_idx in range(output_2d_array_size):
                output_2d_array[row_idx][col_idx] = costmap_data_element[row_idx*output_2d_array_size + col_idx]
        
        shape_size = 5
        slice_start_idx = 0
        output_2d_array = output_2d_array[slice_start_idx:slice_start_idx+shape_size, slice_start_idx:slice_start_idx+shape_size]
        output_2d_array = np.reshape(output_2d_array, (1,shape_size*shape_size))
        print(output_2d_array)
        print(np.shape(output_2d_array))

        imu_sig_numpy = np.array(imu_data[0:100])
        freqs, psd = signal.welch(x=imu_sig_numpy, fs=400.0)

        traversability = 0

        for freq_idx in range(len(freqs)):
            if 0 <= freqs[freq_idx] <= 30:
                traversability = traversability + psd[freq_idx]
        
        with open("data_c_n_r.csv", "ab") as fr:
            np.savetxt(fr,output_2d_array)

        with open("data_c_n_b.csv", "ab") as fb:
            np.savetxt(fb,output_2d_array)

        with open("data_c_n_s.csv", "ab") as fs:
            np.savetxt(fs,output_2d_array)
        
        file_writer.writerow([traversability])

        count = count + 1

    print("Data length:", count)

def raw_data_parser(args):
    coreset = CoresetBuffer(args.coreset_buffer_size, args.c_r_class_num)
    f_c_n_s = np.loadtxt(os.path.join(args.dataset_dir, "data_c_n_s.csv"))
    f_c_n_r = np.loadtxt(os.path.join(args.dataset_dir, "data_c_n_r.csv"))
    f_c_n_b = np.loadtxt(os.path.join(args.dataset_dir, "data_c_n_b.csv"))

    f_c_r = open(os.path.join(args.dataset_dir, "data_c_r.csv"))
    rdr = csv.reader(f_c_r)
    count = 0
    new_ipt = None
    new_opt = None

    for line in rdr:
        tmp_ipt_s = f_c_n_s[count]
        tmp_ipt_r = f_c_n_r[count]
        tmp_ipt_b = f_c_n_b[count]
        
        tmp_ipt_s = np.reshape(tmp_ipt_s, (args.c_n_grid_size, args.c_n_grid_size))
        tmp_ipt_r = np.reshape(tmp_ipt_r, (args.c_n_grid_size, args.c_n_grid_size))
        tmp_ipt_b = np.reshape(tmp_ipt_b, (args.c_n_grid_size, args.c_n_grid_size))
        
        tmp_ipt = np.array([tmp_ipt_s, tmp_ipt_r, tmp_ipt_b])
        
        tmp_opt = line[0]
        coreset.append(tmp_ipt, tmp_opt)
        count = count + 1

        new_ipt = tmp_ipt
        new_opt = tmp_opt
    
    return coreset, new_ipt, new_opt

if __name__ == '__main__':
    main()