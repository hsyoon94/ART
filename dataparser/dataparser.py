#!/usr/bin/env python3
import sys
sys.path.append('../algo')


import numpy as np
import rosbag
from std_msgs.msg import Int32, String
from matplotlib import pyplot as plt
from scipy import signal
import math
import csv
import os


bag = rosbag.Bag('/media/hsyoon94/HS_SDD/bagfiles/230201/visky_2023-02-01-15-34-49.bag')

odom_topic = '/aft_mapped_to_init'
costmap_topic = '/traversability_costmap_visualize'
imu_topic = '/imu/data'

costmap_data = list()
odom_data = list()
imu_data = list()
SHAPE_SIZE = 10
MAP_GRID_LENGTH = 0

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

    timestep = 0
    RECORDING_START = True
    frequency_for_every_quatsec_cost_map = 1
    frequency_for_every_quatsec_cost_odom = 3
    frequency_for_every_quatsec_cost_imu = 100
    
    for timestep_for_quatsec in range(int(len_imu/400)):
        costmap_idx = timestep_for_quatsec * frequency_for_every_quatsec_cost_map # 0, 1, 2, ...
        odom_cur_idx = timestep_for_quatsec * frequency_for_every_quatsec_cost_odom # 0, 3, 6, 9, ...
        odom_fut_idx = (timestep_for_quatsec + 1) * frequency_for_every_quatsec_cost_odom # 0, 3, 6, 9, ...
        imu_idx = timestep_for_quatsec * frequency_for_every_quatsec_cost_imu # 0, 100, 200, 300, ...
        
        costmap_data_element = costmap_data[costmap_idx]
        input_1d_array_len = len(costmap_data_element)
        
        full_output_2d_array_size = int(math.sqrt(input_1d_array_len))
        print("full_output_2d_array_size", full_output_2d_array_size)
        MAP_GRID_LENGTH = 20 / full_output_2d_array_size
        
        output_2d_array = np.zeros((full_output_2d_array_size, full_output_2d_array_size))

        for row_idx in range(full_output_2d_array_size):
            for col_idx in range(full_output_2d_array_size):
                output_2d_array[row_idx][col_idx] = costmap_data_element[row_idx*full_output_2d_array_size + col_idx]
        

        # Convert global pose (from odometry) to local pose in costmap
        origin_x = odom_data[odom_cur_idx].pose.pose.position.x
        origin_y = odom_data[odom_fut_idx].pose.pose.position.y

        cur_x_global = odom_data[odom_fut_idx].pose.pose.position.x
        cur_y_global = odom_data[odom_fut_idx].pose.pose.position.y

        cur_x_local = int((cur_x_global - origin_x) / full_output_2d_array_size) + int(full_output_2d_array_size / 2)
        cur_y_local = int((cur_y_global - origin_y) / full_output_2d_array_size) + int(full_output_2d_array_size / 2)
        
        # Get wheel local pose (ToDo: consider rotation)              
        wheel_lf_x = cur_x_local + int(0.01 / MAP_GRID_LENGTH)
        wheel_lf_y = cur_y_local - int(0.02 / MAP_GRID_LENGTH)

        wheel_rf_x = cur_x_local + int(0.01 / MAP_GRID_LENGTH)
        wheel_rf_y = cur_y_local + int(0.02 / MAP_GRID_LENGTH)

        wheel_lr_x = cur_x_local - int(0.01 / MAP_GRID_LENGTH)
        wheel_lr_y = cur_y_local - int(0.02 / MAP_GRID_LENGTH)

        wheel_rr_x = cur_x_local - int(0.01 / MAP_GRID_LENGTH)
        wheel_rr_y = cur_y_local + int(0.02 / MAP_GRID_LENGTH)

        patch_lf = output_2d_array[wheel_lf_x-int(SHAPE_SIZE / (2 * 2)):wheel_lf_x+int(SHAPE_SIZE / (2 * 2)) + 1, wheel_lf_y-int(SHAPE_SIZE / (2 * 2)):wheel_lf_y+int(SHAPE_SIZE / (2 * 2)) + 1]
        patch_rf = output_2d_array[wheel_rf_x-int(SHAPE_SIZE / (2 * 2)):wheel_rf_x+int(SHAPE_SIZE / (2 * 2)) + 1, wheel_rf_y-int(SHAPE_SIZE / (2 * 2)):wheel_rf_y+int(SHAPE_SIZE / (2 * 2)) + 1]

        patch_lr = output_2d_array[wheel_lr_x-int(SHAPE_SIZE / (2 * 2)):wheel_lr_x+int(SHAPE_SIZE / (2 * 2)) + 1, wheel_lr_y-int(SHAPE_SIZE / (2 * 2)):wheel_lr_y+int(SHAPE_SIZE / (2 * 2)) + 1]
        patch_rr = output_2d_array[wheel_rr_x-int(SHAPE_SIZE / (2 * 2)):wheel_rr_x+int(SHAPE_SIZE / (2 * 2)) + 1, wheel_rr_y-int(SHAPE_SIZE / (2 * 2)):wheel_rr_y+int(SHAPE_SIZE / (2 * 2)) + 1]

        final_patch=np.zeros((SHAPE_SIZE, SHAPE_SIZE))
        final_patch[0:int(SHAPE_SIZE/2), 0:int(SHAPE_SIZE/2)] = patch_lf
        final_patch[int(SHAPE_SIZE/2):SHAPE_SIZE, 0:int(SHAPE_SIZE/2)] = patch_rf
        final_patch[0:int(SHAPE_SIZE/2), int(SHAPE_SIZE/2):SHAPE_SIZE] = patch_lr
        final_patch[int(SHAPE_SIZE/2):SHAPE_SIZE, int(SHAPE_SIZE/2):SHAPE_SIZE] = patch_rr

        print("patch_lf", patch_lf)

        final_patch = np.reshape(final_patch,(1,100))
        
        imu_sig_numpy = np.array(imu_data[imu_idx:imu_idx+frequency_for_every_quatsec_cost_imu])
        freqs, psd = signal.welch(x=imu_sig_numpy, fs=400.0)

        # plt.semilogy(freqs, psd)
        # plt.xlabel('frequency [Hz]')
        # plt.ylabel('PSD')
        # plt.show()

        traversability = 0

        for freq_idx in range(len(freqs)):
            if 0 <= freqs[freq_idx] <= 30:
                traversability = traversability + psd[freq_idx]
        print("traversability", traversability)

        with open("data_c_n_r.csv", "ab") as fr:
            np.savetxt(fr,final_patch)

        with open("data_c_n_b.csv", "ab") as fb:
            np.savetxt(fb,final_patch)

        with open("data_c_n_s.csv", "ab") as fs:
            np.savetxt(fs,final_patch)
        
        file_writer.writerow([traversability])
        
        timestep = timestep + 1
    print("Data length:", timestep)

def raw_data_parser(args):

    from algo.coreset import DatasetBuffer

    coreset = DatasetBuffer(100000, args.c_r_class_num)
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
    
    return coreset

if __name__ == '__main__':
    main()