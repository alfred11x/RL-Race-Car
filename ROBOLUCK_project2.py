#!/usr/bin/env python2
from __future__ import print_function


##### add python path #####
import sys
import os
import rospkg
import rospy

PATH = rospkg.RosPack().get_path("sim2real") + "/scripts"
print(PATH)
sys.path.append(PATH)


import gym
import env
import numpy as np
from collections import deque
import json
import random
import math
import yaml
import time
from sim2real.msg import Result, Query
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
#
from sklearn.gaussian_process.kernels import DotProduct as D
from joblib import dump, load


project_path = rospkg.RosPack().get_path("sim2real")
yaml_file = project_path + "/config/eval1.yaml"

######################################## PLEASE CHANGE TEAM NAME ########################################
TEAM_NAME = "ROBOLUCK"
######################################## PLEASE CHANGE TEAM NAME ########################################
team_path = project_path + "/project/IS_" + TEAM_NAME

class GaussianProcess:
    def __init__(self, args):
        rospy.init_node('gaussian_process_' + TEAM_NAME, anonymous=True, disable_signals=True)
        # env reset with world file
        self.env = gym.make('RCCar-v0')
        self.env.seed(1)
        self.env.unwrapped
        self.env.load(args['world_name'])
        self.track_list = self.env.track_list
        
        self.time_limit = 100.0
        
        """
        add your demonstration files with expert state-action pairs.
        you can collect expert demonstration using pure pursuit.
        you can define additional class member variables.
        """
        #DON'T CHANGE THIS PART!
        # 1.5 <= minVel <= maxVel <= 3.0
        self.maxAng = 1.5
        self.minVel = 0.5
        self.maxVel = 3.0
        ########################
        
        self.demo_files = ['data1.csv']#, 'data2.csv', 'data3.csv', 'data4.csv', 'data5.csv', 'data6.csv', 'data7.csv', 'data8.csv', 'data9.csv', 'data10.csv']#, 'data11.csv', 'data12.csv', 'data13.csv', 'data14.csv', 'data15.csv', 'data16.csv', 'data17.csv', 'data18.csv', 'data19.csv', 'data20.csv', 'data21.csv', 'data22.csv', 'data23.csv', 'data24.csv', 'data25.csv', 'data26.csv', 'data27.csv', 'data28.csv', 'data29.csv', 'data30.csv', 'data31.csv', 'data32.csv', 'data33.csv', 'data34.csv', 'data35.csv', 'data36.csv', 'data37.csv', 'data38.csv', 'data39.csv', 'data40.csv']

        self.obs_num = 1000

        self.demo_obs = []
        self.demo_act = []
	######################## from here
        #self.kernel_steer = RBF(length_scale = 1.0, length_scale_bounds = (0.6, 1000))
	self.kernel_steer = D()
	#self.kernel_vel = RBF(1.0) + C(constant_value=2)
        
        self.gp_steer = GaussianProcessRegressor(kernel = self.kernel_steer)
	#self.gp_vel = GaussianProcessRegressor(kernel = self.kernel_vel)


        self.gp_file_name = "demo"
        self.gp_file = project_path + "/" + self.gp_file_name + ".joblib"
	
	######################## to here
        self.load()

        self.query_sub = rospy.Subscriber("/query", Query, self.callback_query)
        self.rt_pub = rospy.Publisher("/result", Result, queue_size = 1)
	
	######################## from here
	num = self.num
	X = self.normdata[:, :num]
	y = self.normdata[:, -1]
	self.gp_steer.fit(X, y)
	#self.gp_steer = load(team_path + "/project/model.joblib")
	#self.gp_vel.fit(X, y_vel)
	
	######################## to here
	self.i = 0
		
	
        print("completed initialization")


    def load(self):
        """
        1) load your expert demonstration
        2) normalization and gp fitting (recommand using scikit-learn package)
        3) if you already have a fitted model, load it instead of fitting the model again.
        Please implement loading pretrained model part for fast evaluation.
        """
	#expert_data_PATH = 'catkin_ws/src/Intelligent-Systems-2023/sim2real/project/IS_ROBOLUCK/project/'
	expert_data_PATH = team_path + '/project/'
	expert_data = np.loadtxt(expert_data_PATH + self.demo_files[0], delimiter=',')
	steers = expert_data[:400, -1].reshape(-1, 1)
	if len(self.demo_files) > 1:
	    for i in range(1, len(self.demo_files)):
		new_data = np.loadtxt(expert_data_PATH + self.demo_files[i], delimiter=',')
		expert_data = np.concatenate((expert_data, new_data), axis=0)
		steers = np.concatenate((steers, new_data[:500, -1]), axis=1)

	num = 5
	self.num = num
	data = np.zeros((expert_data.shape[0], num+1))
	data[:, :num] = expert_data[:, :num]
	data[:, num] = expert_data[:, -1]
	self.means = np.mean(data, axis=0)
	self.stds = np.std(data, axis=0)
	self.normdata = (data - self.means) / self.stds
	self.steers = steers
	
	

    def get_action(self, obs):
        """
        1) input observation is the raw data from environment.
        2) 0 to 1080 components (totally 1081) are from lidar.
           Also, 1081 component and 1082 component are scaled velocity and steering angle, respectively.
        3) To improve your algorithm, you must process the raw data so that the model fits well.
        4) Always keep in mind of normalization process during implementation.
        """
	'''
	num = self.num
	temp = obs
	obs = np.zeros((1, num+1))
	obs[:, :num] = temp[:, :num]
	obs[:, num] = temp[:, -1]
	normobs = (obs - self.means) / self.stds
	x = normobs[:1, :num]
	s = self.gp_steer.predict(x) * self.stds[-1] + self.means[-1]
	'''
	print(self.steers.shape)
	s = self.steers[self.i, :]#.mean()
	self.i = self.i + 1
	return np.array([s, np.array([1.5])]).transpose()

    def callback_query(self, data):
        rt = Result()
        START_TIME = time.time()
        is_exit = data.exit
        try:
            # if query is valid, start
            if data.name != TEAM_NAME:
                return
            
            if data.world not in self.track_list:
                END_TIME = time.time()
                rt.id = data.id
                rt.trial = data.trial
                rt.team = data.name
                rt.world = data.world
                rt.elapsed_time = END_TIME - START_TIME
                rt.waypoints = 0
                rt.n_waypoints = 20
                rt.success = False
                rt.fail_type = "Invalid Track"
                self.rt_pub.publish(rt)
                return
            
            print("[%s] START TO EVALUATE! MAP NAME: %s" %(data.name, data.world))
            obs = self.env.reset(name = data.world)
            obs = np.reshape(obs, [1,-1])
	    print("obs:", obs[:, :10])
            #
	    expert_data = obs
            
            while True:
                if time.time() - START_TIME > self.time_limit:
                    END_TIME = time.time()
                    rt.id = data.id
                    rt.trial = data.trial
                    rt.team = data.name
                    rt.world = data.world
                    rt.elapsed_time = END_TIME - START_TIME
                    rt.waypoints = self.env.next_checkpoint
                    rt.n_waypoints = 20
                    rt.success = False
                    rt.fail_type = "Exceed Time Limit"
                    self.rt_pub.publish(rt)
                    print("EXCEED TIME LIMIT")
                    break
                
                act = self.get_action(obs)
                input_steering = np.clip(act[0][0], -self.maxAng, self.maxAng)
                input_velocity = np.clip(act[0][1], self.minVel, self.maxVel)
                obs, _, done, logs = self.env.step([input_steering, input_velocity])
                obs = np.reshape(obs, [1,-1])
		#
		expert_data = np.concatenate((expert_data, obs), axis=0)
                
                if done:
                    END_TIME = time.time()
                    rt.id = data.id
                    rt.trial = data.trial
                    rt.team = data.name
                    rt.world = data.world
                    rt.elapsed_time = END_TIME - START_TIME
                    rt.waypoints = logs['checkpoints']
                    rt.n_waypoints = 20
                    rt.success = True if logs['info'] == 3 else False
                    rt.fail_type = ""
                    print(logs)
                    if logs['info'] == 1:
                        rt.fail_type = "Collision"
                    if logs['info'] == 2:
                        rt.fail_type = "Exceed Time Limit"
                    self.rt_pub.publish(rt)
                    print("publish result")
                    break
        
        except Exception as e:
            print(e)
            END_TIME = time.time()
            rt.id = data.id
            rt.trial = data.trial
            rt.team = data.name
            rt.world = data.world
            rt.elapsed_time = END_TIME - START_TIME
            rt.waypoints = 0
            rt.n_waypoints = 20
            rt.success = False
            rt.fail_type = "Script Error"
            self.rt_pub.publish(rt)
	#np.savetxt(project_path + '/project/IS_ROBOLUCK/project/sim1.csv', expert_data, delimiter=",")
	#print("Successfully saved the expert data")
        if is_exit:
            rospy.signal_shutdown("End query")
        
        return

if __name__ == '__main__':
    with open(yaml_file) as file:
        args = yaml.load(file)
    GaussianProcess(args)
    rospy.spin()
