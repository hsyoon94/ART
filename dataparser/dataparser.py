#!/usr/bin/env python3

import numpy as np
import rosbag
from std_msgs.msg import Int32, String
from matplotlib import pyplot as plt
from scipy import signal
import math
import csv

bag = rosbag.Bag('/media/hsyoon94/HS_SDD/bagfiles/230201/visky_2023-02-01-15-34-49.bag')

odom_topic = '/aft_mapped_to_init'
costmap_topic = '/traversability_costmap_visualize'
imu_topic = '/imu/data'

costmap_data = list()
odom_data = list()
imu_data = list()

def main():
    
    file_write = open('data.csv', 'w', newline="")
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

    for costmap_data_element in costmap_data:

        input_1d_array_len = len(costmap_data_element)
        
        output_2d_array_size = math.sqrt(input_1d_array_len)
        output_2d_array_size = int(output_2d_array_size)
        output_2d_array = np.zeros((output_2d_array_size, output_2d_array_size))

        for row_idx in range(output_2d_array_size):
            for col_idx in range(output_2d_array_size):
                output_2d_array[row_idx][col_idx] = costmap_data_element[row_idx*output_2d_array_size + col_idx]
        
        output_2d_array = output_2d_array[0:10, 0:10]

        imu_sig_numpy = np.array(imu_data[0:100])
        freqs, psd = signal.welch(x=imu_sig_numpy, fs=400.0)

        traversability = 0

        for freq_idx in range(len(freqs)):
            if 0 <= freqs[freq_idx] <= 30:
                traversability = traversability + psd[freq_idx]
        
        # plt.figure(figsize=(5, 4))
        # plt.semilogx(freqs, psd)
        # plt.title('PSD: power spectral density')
        # plt.xlabel('Frequency')
        # plt.ylabel('Power')
        # plt.tight_layout()
        # plt.show()

        print("c_g", output_2d_array)
        print("c_r", traversability)
        
        file_writer.writerow([output_2d_array, traversability])
    
if __name__ == '__main__':
    main()