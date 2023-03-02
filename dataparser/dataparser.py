#!/usr/bin/env python3

import numpy as np
import rosbag
from std_msgs.msg import Int32, String
from matplotlib import pyplot as plt
from scipy import signal

bag = rosbag.Bag('/media/hsyoon94/HS_SDD/bagfiles/230201/visky_2023-02-01-15-34-49.bag')

odom_topic = '/aft_mapped_to_init'
costmap_topic = '/traversability_costmap'
imu_topic = '/imu/data'

costmap_data = list()
odom_data = list()
imu_data = list()

for topic, msg, t in bag.read_messages(topics=[costmap_topic, odom_topic, imu_topic]):
    if topic == costmap_topic:
        costmap_data.append(msg)
    
    elif topic == odom_topic:

        # print(msg.twist.twist.linear.x)
        odom_data.append(msg)

    elif topic == imu_topic:
        imu_data.append(msg)

print(len(costmap_data))
print(len(odom_data))
print(len(imu_data))

practice = np.zeros((5, 5))

np.savetxt("example.txt", practice)