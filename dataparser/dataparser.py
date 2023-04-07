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
from PIL import Image
import random

# roslaunch visky visky.launch
# rosbag record -o bag_for_eval /imu/data /aft_mapped_to_init /traversability_costmap_roughness /traversability_costmap_slippage /traversability_costmap_slope

bag = rosbag.Bag('/home/hsyoon94/bagfiles/bag_for_eval_2023-03-16-10-47-33.bag')

odom_topic = '/aft_mapped_to_init'
costmap_topic_roughness = '/traversability_costmap_roughness'
costmap_topic_slippage = '/traversability_costmap_slippage'
costmap_topic_slope = '/traversability_costmap_slope'
imu_topic = '/imu/data'

costmap_data_roughness = list()
costmap_data_slippage = list()
costmap_data_slope = list()
odom_data = list()
imu_data = list()
SHAPE_SIZE = 10
MAP_GRID_LENGTH = 0
CLASS_NUM = 10

def main():
    
    remove_old_files()
    file_write = open('data_c_r.csv', 'w', newline="")
    file_write_c3 = open('data_c_r_c3.csv', 'w', newline="")
    file_write_c5 = open('data_c_r_c5.csv', 'w', newline="")
    file_write_c10 = open('data_c_r_c10.csv', 'w', newline="")

    file_write_eval = open('data_c_r_eval.csv', 'w', newline="")
    file_write_c3_eval = open('data_c_r_c3_eval.csv', 'w', newline="")
    file_write_c5_eval = open('data_c_r_c5_eval.csv', 'w', newline="")
    file_write_c10_eval = open('data_c_r_c10_eval.csv', 'w', newline="")
    raw_traversability_list = list()
    file_writer = csv.writer(file_write)
    file_writer_c3 = csv.writer(file_write_c3)
    file_writer_c5 = csv.writer(file_write_c5)
    file_writer_c10 = csv.writer(file_write_c10)

    file_writer_eval = csv.writer(file_write_eval)
    file_writer_c3_eval = csv.writer(file_write_c3_eval)
    file_writer_c5_eval = csv.writer(file_write_c5_eval)
    file_writer_c10_eval = csv.writer(file_write_c10_eval)

    trav_min = 11111
    trav_max = 0

    occupancy_grid = np.array([])
    
    for topic, msg, t in bag.read_messages(topics=[costmap_topic_roughness, costmap_topic_slippage, costmap_topic_slope, odom_topic, imu_topic]):
        if topic == costmap_topic_roughness:
            costmap_data_roughness.append(msg.data)

        elif topic == costmap_topic_slippage:
            costmap_data_slippage.append(msg.data)

        elif topic == costmap_topic_slope:
            costmap_data_slope.append(msg.data)

        elif topic == odom_topic:
            odom_data.append(msg)

        elif topic == imu_topic:
            imu_data.append(msg.orientation.z)

    len_imu = len(imu_data)
    len_cost = len(costmap_data_slope)
    len_odom = len(odom_data)
    
    real_time_duration_second = len_imu / 400

    timestep = 0
    RECORDING_START = True
    frequency_for_every_quatsec_cost_map = (len_cost/real_time_duration_second)/4
    frequency_for_every_quatsec_cost_odom = (len_odom/real_time_duration_second)/4
    frequency_for_every_quatsec_cost_imu = 100
    
    LAST_TWO_COUNT = int(len_imu/(400/4))
    traversability_class3_count = np.zeros(shape=(3,))
    traversability_class5_count = np.zeros(shape=(5,))
    traversability_class10_count = np.zeros(shape=(10,))

    for timestep_for_quatsec in range(int(len_imu/(400/4))):
        costmap_idx = int(timestep_for_quatsec * frequency_for_every_quatsec_cost_map) # 0, 1, 2, ...
        odom_cur_idx = int(timestep_for_quatsec * frequency_for_every_quatsec_cost_odom) # 0, 3, 6, 9, ...
        odom_fut_idx = int((timestep_for_quatsec + 1) * frequency_for_every_quatsec_cost_odom) # 0, 3, 6, 9, ...
        imu_idx = timestep_for_quatsec * frequency_for_every_quatsec_cost_imu # 0, 100, 200, 300, ...
        
        costmap_data_roughness_element = costmap_data_roughness[costmap_idx]
        costmap_data_slippage_element = costmap_data_slippage[costmap_idx]
        costmap_data_slope_element = costmap_data_slope[costmap_idx]

        input_1d_array_len = len(costmap_data_roughness_element)
        
        full_output_2d_array_size = int(math.sqrt(input_1d_array_len))

        MAP_GRID_LENGTH = 20 / full_output_2d_array_size
        
        output_2d_array_roughness = np.zeros((full_output_2d_array_size, full_output_2d_array_size))
        output_2d_array_slippage = np.zeros((full_output_2d_array_size, full_output_2d_array_size))
        output_2d_array_slope = np.zeros((full_output_2d_array_size, full_output_2d_array_size))

        for row_idx in range(full_output_2d_array_size):
            for col_idx in range(full_output_2d_array_size):
                output_2d_array_roughness[row_idx][col_idx] = costmap_data_roughness_element[row_idx*full_output_2d_array_size + col_idx]
                output_2d_array_slippage[row_idx][col_idx] = costmap_data_slippage_element[row_idx*full_output_2d_array_size + col_idx]
                output_2d_array_slope[row_idx][col_idx] = costmap_data_slope_element[row_idx*full_output_2d_array_size + col_idx]
        
        # Coordinate
        wheel_lf_x = int(full_output_2d_array_size/2 - (odom_data[odom_fut_idx].pose.pose.position.y-odom_data[odom_cur_idx].pose.pose.position.y)/MAP_GRID_LENGTH - 1*math.sin(odom_data[odom_fut_idx].pose.pose.orientation.x) - 2*math.cos(odom_data[odom_fut_idx].pose.pose.orientation.x))
        wheel_lf_y = int(full_output_2d_array_size/2 + (odom_data[odom_fut_idx].pose.pose.position.x-odom_data[odom_cur_idx].pose.pose.position.x)/MAP_GRID_LENGTH - 1*math.cos(odom_data[odom_fut_idx].pose.pose.orientation.x) + 2*math.sin(odom_data[odom_fut_idx].pose.pose.orientation.x))

        wheel_rf_x = int(full_output_2d_array_size/2 - (odom_data[odom_fut_idx].pose.pose.position.y-odom_data[odom_cur_idx].pose.pose.position.y)/MAP_GRID_LENGTH + 1*math.sin(odom_data[odom_fut_idx].pose.pose.orientation.x) - 2*math.cos(odom_data[odom_fut_idx].pose.pose.orientation.x))
        wheel_rf_y = int(full_output_2d_array_size/2 + (odom_data[odom_fut_idx].pose.pose.position.x-odom_data[odom_cur_idx].pose.pose.position.x)/MAP_GRID_LENGTH + 1*math.cos(odom_data[odom_fut_idx].pose.pose.orientation.x) + 2*math.sin(odom_data[odom_fut_idx].pose.pose.orientation.x))

        wheel_lr_x = int(full_output_2d_array_size/2 - (odom_data[odom_fut_idx].pose.pose.position.y-odom_data[odom_cur_idx].pose.pose.position.y)/MAP_GRID_LENGTH - 1*math.sin(odom_data[odom_fut_idx].pose.pose.orientation.x) + 2*math.cos(odom_data[odom_fut_idx].pose.pose.orientation.x))
        wheel_lr_y = int(full_output_2d_array_size/2 + (odom_data[odom_fut_idx].pose.pose.position.x-odom_data[odom_cur_idx].pose.pose.position.x)/MAP_GRID_LENGTH - 1*math.cos(odom_data[odom_fut_idx].pose.pose.orientation.x) - 2*math.sin(odom_data[odom_fut_idx].pose.pose.orientation.x))

        wheel_rr_x = int(full_output_2d_array_size/2 - (odom_data[odom_fut_idx].pose.pose.position.y-odom_data[odom_cur_idx].pose.pose.position.y)/MAP_GRID_LENGTH + 1*math.sin(odom_data[odom_fut_idx].pose.pose.orientation.x) + 2*math.cos(odom_data[odom_fut_idx].pose.pose.orientation.x))
        wheel_rr_y = int(full_output_2d_array_size/2 + (odom_data[odom_fut_idx].pose.pose.position.x-odom_data[odom_cur_idx].pose.pose.position.x)/MAP_GRID_LENGTH + 1*math.cos(odom_data[odom_fut_idx].pose.pose.orientation.x) - 2*math.sin(odom_data[odom_fut_idx].pose.pose.orientation.x))

        # Roughness
        patch_lf_roughness = output_2d_array_roughness[wheel_lf_x-int(SHAPE_SIZE / (2 * 2)):wheel_lf_x+int(SHAPE_SIZE / (2 * 2)) + 1, wheel_lf_y-int(SHAPE_SIZE / (2 * 2)):wheel_lf_y+int(SHAPE_SIZE / (2 * 2)) + 1]
        patch_rf_roughness = output_2d_array_roughness[wheel_rf_x-int(SHAPE_SIZE / (2 * 2)):wheel_rf_x+int(SHAPE_SIZE / (2 * 2)) + 1, wheel_rf_y-int(SHAPE_SIZE / (2 * 2)):wheel_rf_y+int(SHAPE_SIZE / (2 * 2)) + 1]
        patch_lr_roughness = output_2d_array_roughness[wheel_lr_x-int(SHAPE_SIZE / (2 * 2)):wheel_lr_x+int(SHAPE_SIZE / (2 * 2)) + 1, wheel_lr_y-int(SHAPE_SIZE / (2 * 2)):wheel_lr_y+int(SHAPE_SIZE / (2 * 2)) + 1]
        patch_rr_roughness = output_2d_array_roughness[wheel_rr_x-int(SHAPE_SIZE / (2 * 2)):wheel_rr_x+int(SHAPE_SIZE / (2 * 2)) + 1, wheel_rr_y-int(SHAPE_SIZE / (2 * 2)):wheel_rr_y+int(SHAPE_SIZE / (2 * 2)) + 1]

        final_patch_roughness =np.zeros((SHAPE_SIZE, SHAPE_SIZE))
        final_patch_roughness[0:int(SHAPE_SIZE/2), 0:int(SHAPE_SIZE/2)] = patch_lf_roughness
        final_patch_roughness[int(SHAPE_SIZE/2):SHAPE_SIZE, 0:int(SHAPE_SIZE/2)] = patch_rf_roughness
        final_patch_roughness[0:int(SHAPE_SIZE/2), int(SHAPE_SIZE/2):SHAPE_SIZE] = patch_lr_roughness
        final_patch_roughness[int(SHAPE_SIZE/2):SHAPE_SIZE, int(SHAPE_SIZE/2):SHAPE_SIZE] = patch_rr_roughness
        final_patch_roughness = np.reshape(final_patch_roughness,(1,SHAPE_SIZE * SHAPE_SIZE))

        # Slope
        patch_lf_slope = output_2d_array_slope[wheel_lf_x-int(SHAPE_SIZE / (2 * 2)):wheel_lf_x+int(SHAPE_SIZE / (2 * 2)) + 1, wheel_lf_y-int(SHAPE_SIZE / (2 * 2)):wheel_lf_y+int(SHAPE_SIZE / (2 * 2)) + 1]
        patch_rf_slope = output_2d_array_slope[wheel_rf_x-int(SHAPE_SIZE / (2 * 2)):wheel_rf_x+int(SHAPE_SIZE / (2 * 2)) + 1, wheel_rf_y-int(SHAPE_SIZE / (2 * 2)):wheel_rf_y+int(SHAPE_SIZE / (2 * 2)) + 1]
        patch_lr_slope = output_2d_array_slope[wheel_lr_x-int(SHAPE_SIZE / (2 * 2)):wheel_lr_x+int(SHAPE_SIZE / (2 * 2)) + 1, wheel_lr_y-int(SHAPE_SIZE / (2 * 2)):wheel_lr_y+int(SHAPE_SIZE / (2 * 2)) + 1]
        patch_rr_slope = output_2d_array_slope[wheel_rr_x-int(SHAPE_SIZE / (2 * 2)):wheel_rr_x+int(SHAPE_SIZE / (2 * 2)) + 1, wheel_rr_y-int(SHAPE_SIZE / (2 * 2)):wheel_rr_y+int(SHAPE_SIZE / (2 * 2)) + 1]

        final_patch_slope =np.zeros((SHAPE_SIZE, SHAPE_SIZE))
        final_patch_slope[0:int(SHAPE_SIZE/2), 0:int(SHAPE_SIZE/2)] = patch_lf_slope
        final_patch_slope[int(SHAPE_SIZE/2):SHAPE_SIZE, 0:int(SHAPE_SIZE/2)] = patch_rf_slope
        final_patch_slope[0:int(SHAPE_SIZE/2), int(SHAPE_SIZE/2):SHAPE_SIZE] = patch_lr_slope
        final_patch_slope[int(SHAPE_SIZE/2):SHAPE_SIZE, int(SHAPE_SIZE/2):SHAPE_SIZE] = patch_rr_slope
        final_patch_slope = np.reshape(final_patch_slope,(1,SHAPE_SIZE * SHAPE_SIZE))

        # Slippage
        patch_lf_slippage = output_2d_array_slippage[wheel_lf_x-int(SHAPE_SIZE / (2 * 2)):wheel_lf_x+int(SHAPE_SIZE / (2 * 2)) + 1, wheel_lf_y-int(SHAPE_SIZE / (2 * 2)):wheel_lf_y+int(SHAPE_SIZE / (2 * 2)) + 1]
        patch_rf_slippage = output_2d_array_slippage[wheel_rf_x-int(SHAPE_SIZE / (2 * 2)):wheel_rf_x+int(SHAPE_SIZE / (2 * 2)) + 1, wheel_rf_y-int(SHAPE_SIZE / (2 * 2)):wheel_rf_y+int(SHAPE_SIZE / (2 * 2)) + 1]
        patch_lr_slippage = output_2d_array_slippage[wheel_lr_x-int(SHAPE_SIZE / (2 * 2)):wheel_lr_x+int(SHAPE_SIZE / (2 * 2)) + 1, wheel_lr_y-int(SHAPE_SIZE / (2 * 2)):wheel_lr_y+int(SHAPE_SIZE / (2 * 2)) + 1]
        patch_rr_slippage = output_2d_array_slippage[wheel_rr_x-int(SHAPE_SIZE / (2 * 2)):wheel_rr_x+int(SHAPE_SIZE / (2 * 2)) + 1, wheel_rr_y-int(SHAPE_SIZE / (2 * 2)):wheel_rr_y+int(SHAPE_SIZE / (2 * 2)) + 1]

        final_patch_slippage =np.zeros((SHAPE_SIZE, SHAPE_SIZE))
        final_patch_slippage[0:int(SHAPE_SIZE/2), 0:int(SHAPE_SIZE/2)] = patch_lf_slippage
        final_patch_slippage[int(SHAPE_SIZE/2):SHAPE_SIZE, 0:int(SHAPE_SIZE/2)] = patch_rf_slippage
        final_patch_slippage[0:int(SHAPE_SIZE/2), int(SHAPE_SIZE/2):SHAPE_SIZE] = patch_lr_slippage
        final_patch_slippage[int(SHAPE_SIZE/2):SHAPE_SIZE, int(SHAPE_SIZE/2):SHAPE_SIZE] = patch_rr_slippage
        final_patch_slippage = np.reshape(final_patch_slippage,(1,SHAPE_SIZE * SHAPE_SIZE))

        # Robot-side Traversability
        imu_sig_numpy = np.array(imu_data[imu_idx:imu_idx+frequency_for_every_quatsec_cost_imu])
        freqs, psd = signal.welch(x=imu_sig_numpy, fs=400.0)

        traversability = 0
        scale = 1e8
        traversability_noise_threshold = 300
        # Traversability regression
        for freq_idx in range(len(freqs)):
            if 0 <= freqs[freq_idx] <= 30:
                traversability = traversability + psd[freq_idx]
        traversability = traversability * scale
        
        if traversability <= traversability_noise_threshold:
            raw_traversability_list.append(traversability)

            trav_min = min(traversability, trav_min)
            trav_max = max(traversability, trav_max)

            print("trav min", trav_min, "trav_max", trav_max)

            # Traversability classification
            trav3_list_idx = traversability / 3
            trav3_list_idx = min(trav3_list_idx, 2)
            
            traversability_class3 = np.zeros(shape=(3,))
            traversability_class3[int(trav3_list_idx)] = 1
            traversability_class3_count[int(trav3_list_idx)] = traversability_class3_count[int(trav3_list_idx)] + 1

            trav5_list_idx = traversability / 5
            trav5_list_idx = min(trav5_list_idx, 4)
            
            traversability_class5 = np.zeros(shape=(5,))
            traversability_class5[int(trav5_list_idx)] = 1
            traversability_class5_count[int(trav5_list_idx)] = traversability_class5_count[int(trav5_list_idx)] + 1

            trav10_list_idx = traversability / 10
            trav10_list_idx = min(trav10_list_idx, 9)
            
            traversability_class10 = np.zeros(shape=(10,))
            traversability_class10[int(trav10_list_idx)] = 1
            traversability_class10_count[int(trav10_list_idx)] = traversability_class10_count[int(trav10_list_idx)] + 1

            random_probability = random.uniform(0, 1)
            # For evaluation set
            if 0 <= random_probability <= 0.1:
                with open("data_c_n_r_eval.csv", "ab") as fr:
                    np.savetxt(fr,final_patch_roughness)

                with open("data_c_n_b_eval.csv", "ab") as fb:
                    np.savetxt(fb,final_patch_slope)

                with open("data_c_n_s_eval.csv", "ab") as fs:
                    np.savetxt(fs,final_patch_slippage)
                
                file_writer_eval.writerow([traversability])
                # file_writer_c3_eval.writerow(traversability_class3)
                # file_writer_c5_eval.writerow(traversability_class5)
                # file_writer_c10_eval.writerow(traversability_class10)
                file_writer_c3_eval.writerow([int(trav3_list_idx)])
                file_writer_c5_eval.writerow([int(trav5_list_idx)])
                file_writer_c10_eval.writerow([int(trav10_list_idx)])
            
            else:
                with open("data_c_n_r.csv", "ab") as fr:
                    np.savetxt(fr,final_patch_roughness)

                with open("data_c_n_b.csv", "ab") as fb:
                    np.savetxt(fb,final_patch_slope)

                with open("data_c_n_s.csv", "ab") as fs:
                    np.savetxt(fs,final_patch_slippage)
                
                file_writer.writerow([traversability])
                # file_writer_c3.writerow(traversability_class3)
                # file_writer_c5.writerow(traversability_class5)
                # file_writer_c10.writerow(traversability_class10)
                file_writer_c3.writerow([int(trav3_list_idx)])
                file_writer_c5.writerow([int(trav5_list_idx)])
                file_writer_c10.writerow([int(trav10_list_idx)])

            if (LAST_TWO_COUNT - timestep_for_quatsec) == 2 or (LAST_TWO_COUNT - timestep_for_quatsec) == 1:
                with open("inference/data_c_n_r_inference.csv", "ab") as frinf:
                    np.savetxt(frinf,np.reshape(output_2d_array_roughness, (1, input_1d_array_len)))
                    output_2d_array_roughness = output_2d_array_roughness.astype(np.uint8)
                    im = Image.fromarray(output_2d_array_roughness)
                    if (LAST_TWO_COUNT - timestep_for_quatsec) == 2:
                        im.save("inference/image_roughness_0.jpg")
                    elif (LAST_TWO_COUNT - timestep_for_quatsec) == 1:
                        im.save("inference/image_roughness_1.jpg")

                with open("inference/data_c_n_b_inference.csv", "ab") as fbinf:
                    np.savetxt(fbinf,np.reshape(output_2d_array_slope, (1, input_1d_array_len)))
                    output_2d_array_slope = output_2d_array_slope.astype(np.uint8)
                    im = Image.fromarray(output_2d_array_slope)
                    if (LAST_TWO_COUNT - timestep_for_quatsec) == 2:
                        im.save("inference/image_slope_0.jpg")
                    elif (LAST_TWO_COUNT - timestep_for_quatsec) == 1:
                        im.save("inference/image_slope_1.jpg")

                with open("inference/data_c_n_s_inference.csv", "ab") as fsinf:
                    np.savetxt(fsinf,np.reshape(output_2d_array_slippage, (1, input_1d_array_len)))
                    output_2d_array_slippage = output_2d_array_slippage.astype(np.uint8)
                    im = Image.fromarray(output_2d_array_slippage)
                    if (LAST_TWO_COUNT - timestep_for_quatsec) == 2:
                        im.save("inference/image_slippage_0.jpg")
                    elif (LAST_TWO_COUNT - timestep_for_quatsec) == 1:
                        im.save("inference/image_slippage_1.jpg")
        
        print("Timestep", timestep)
        timestep = timestep + 1

    print("Data length:", timestep)
    print("Data c3 distribution", traversability_class3_count)
    print("Data c5 distribution", traversability_class5_count)
    print("Data c10 distribution", traversability_class10_count)

    plt.plot(raw_traversability_list)
    plt.show()

def remove_old_files():

    if os.path.exists('data_c_r.csv'):
        os.remove('data_c_r.csv')
    if os.path.exists('data_c_r_c3.csv'):
        os.remove('data_c_r_c3.csv')
    if os.path.exists('data_c_r_c5.csv'):
        os.remove('data_c_r_c5.csv')
    if os.path.exists('data_c_r_c10.csv'):
        os.remove('data_c_r_c10.csv')

    if os.path.exists('data_c_n_b.csv'):
        os.remove('data_c_n_b.csv')
    if os.path.exists('data_c_n_s.csv'):
        os.remove('data_c_n_s.csv')
    if os.path.exists('data_c_n_r.csv'):
        os.remove('data_c_n_r.csv')


    if os.path.exists('data_c_r_eval.csv'):
        os.remove('data_c_r_eval.csv')
    if os.path.exists('data_c_r_c3_eval.csv'):
        os.remove('data_c_r_c3_eval.csv')
    if os.path.exists('data_c_r_c5_eval.csv'):
        os.remove('data_c_r_c5_eval.csv')
    if os.path.exists('data_c_r_c10_eval.csv'):
        os.remove('data_c_r_c10_eval.csv')

    if os.path.exists('data_c_n_b_eval.csv'):
        os.remove('data_c_n_b_eval.csv')
    if os.path.exists('data_c_n_s_eval.csv'):
        os.remove('data_c_n_s_eval.csv')
    if os.path.exists('data_c_n_r_eval.csv'):
        os.remove('data_c_n_r_eval.csv')

    if os.path.exists('inference/data_c_n_b_inference.csv'):
        os.remove('inference/data_c_n_b_inference.csv')
    if os.path.exists('inference/data_c_n_s_inference.csv'):
        os.remove('inference/data_c_n_s_inference.csv')
    if os.path.exists('inference/data_c_n_r_inference.csv'):
        os.remove('inference/data_c_n_r_inference.csv')

def raw_data_parser(args):

    from algo.DatasetBuffer import DatasetBuffer

    coreset = DatasetBuffer(100000, args.coreset_type, args.c_r_class_num, args.experiment, "train")
    f_c_n_s = np.loadtxt(os.path.join(args.dataset_dir, "data_c_n_s.csv"))
    f_c_n_r = np.loadtxt(os.path.join(args.dataset_dir, "data_c_n_r.csv"))
    f_c_n_b = np.loadtxt(os.path.join(args.dataset_dir, "data_c_n_b.csv"))

    if args.c_r_class_num == 1:
        f_c_r = open(os.path.join(args.dataset_dir, "data_c_r.csv"))
    elif args.c_r_class_num == 3:
        f_c_r = open(os.path.join(args.dataset_dir, "data_c_r_c3.csv"))
    elif args.c_r_class_num == 5:
        f_c_r = open(os.path.join(args.dataset_dir, "data_c_r_c5.csv"))
    elif args.c_r_class_num == 10:
        f_c_r = open(os.path.join(args.dataset_dir, "data_c_r_c10.csv"))

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
        if args.c_r_class_num == 1:
            tmp_opt = float(line[0])
        else:
            tmp_opt = [float(item) for item in line]
            tmp_opt = np.array(tmp_opt)
        
        coreset.sequantial_append(tmp_ipt, tmp_opt)
        count = count + 1

        new_ipt = tmp_ipt
        new_opt = tmp_opt
    
    return coreset

def eval_data_parser(args):

    from algo.DatasetBuffer import DatasetBuffer
    coreset = DatasetBuffer(100000, args.coreset_type, args.c_r_class_num, args.experiment, "train")
    f_c_n_s = np.loadtxt(os.path.join(args.dataset_dir, "data_c_n_s_eval.csv"))
    f_c_n_r = np.loadtxt(os.path.join(args.dataset_dir, "data_c_n_r_eval.csv"))
    f_c_n_b = np.loadtxt(os.path.join(args.dataset_dir, "data_c_n_b_eval.csv"))

    if args.c_r_class_num == 1:
        f_c_r = open(os.path.join(args.dataset_dir, "data_c_r_eval.csv"))
    elif args.c_r_class_num == 3:
        f_c_r = open(os.path.join(args.dataset_dir, "data_c_r_c3_eval.csv"))
    elif args.c_r_class_num == 5:
        f_c_r = open(os.path.join(args.dataset_dir, "data_c_r_c5_eval.csv"))
    elif args.c_r_class_num == 10:
        f_c_r = open(os.path.join(args.dataset_dir, "data_c_r_c10_eval.csv"))

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
        if args.c_r_class_num == 1:
            tmp_opt = float(line[0])
        else:
            tmp_opt = [float(item) for item in line]
            tmp_opt = np.array(tmp_opt)
        
        coreset.sequantial_append(tmp_ipt, tmp_opt)
        count = count + 1

        new_ipt = tmp_ipt
        new_opt = tmp_opt
    
    return coreset


def inference_data_parser(args):

    from algo.DatasetBuffer import DatasetBuffer

    inference_dataset = DatasetBuffer(1000, args.coreset_type, args.c_r_class_num, args.experiment, "inference")
    f_c_n_s = np.loadtxt(os.path.join(args.dataset_dir, "inference", "data_c_n_s_inference.csv"))
    f_c_n_r = np.loadtxt(os.path.join(args.dataset_dir, "inference", "data_c_n_r_inference.csv"))
    f_c_n_b = np.loadtxt(os.path.join(args.dataset_dir, "inference", "data_c_n_b_inference.csv"))

    count = 0
    new_ipt = None
    new_opt = None

    for count in range(2):
        tmp_ipt_s = f_c_n_s[count]
        tmp_ipt_r = f_c_n_r[count]
        tmp_ipt_b = f_c_n_b[count]
        
        tmp_ipt_s = np.reshape(tmp_ipt_s, (400, 400))
        tmp_ipt_r = np.reshape(tmp_ipt_r, (400, 400))
        tmp_ipt_b = np.reshape(tmp_ipt_b, (400, 400))
        
        tmp_ipt = np.array([tmp_ipt_s, tmp_ipt_r, tmp_ipt_b])
        
        inference_dataset.append(tmp_ipt, None, None, 0)
    
    return inference_dataset

if __name__ == '__main__':
    main()