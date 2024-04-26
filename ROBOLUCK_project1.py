#!/usr/bin/env python2
from __future__ import print_function
import sys
import os
import rospkg
import rospy
######################################## PLEASE CHANGE TEAM NAME ########################################
TEAM_NAME = "ROBOLUCK"
######################################## PLEASE CHANGE TEAM NAME ########################################
project_path = rospkg.RosPack().get_path("sim2real")
yaml_file = project_path + "/config/eval3.yaml"
PATH = project_path + "/scripts"
sys.path.append(PATH)
from sim2real.msg import Result, Query
import gym
import env
import numpy as np
import math
import yaml
import time
def dist(waypoint, pos):
    return math.sqrt((waypoint[0] - pos.x) ** 2 + (waypoint[1] - pos.y) ** 2)
class PurePursuit:
    def __init__(self, args):
        rospy.init_node(TEAM_NAME + "_project1", anonymous=True, disable_signals=True)
        # env reset with world file
        self.env = gym.make('RCCar-v0')
        self.env.seed(1)
        self.env.unwrapped
        self.env.load(args['world_name'])
        self.track_list = self.env.track_list
        self.time_limit = 150.0
        ######################################## YOU CAN ONLY CHANGE THIS PART ########################################
        # TO DO
        """ 
        Setting hyperparameter
        Recommend tuning PID coefficient P->D->I order.
        Also, Recommend set Ki extremely low.
        """
        self.lookahead = 10
        self.prv_err = None
        self.cur_err = None
        self.sum_err = 0.0
        self.dt = 0.1
        self.min_vel = 1.5
        self.max_vel = 2.5
        self.Kp = 2.5
        self.Ki = 0.001
        self.Kd = 1.25
        ######################################## YOU CAN ONLY CHANGE THIS PART ########################################
        self.query_sub = rospy.Subscriber("/query", Query, self.callback_query)
        self.rt_pub = rospy.Publisher("/result", Result, queue_size = 1)
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
            self.env.reset(name = data.world)
            self.waypoints = self.env.waypoints_list[self.env.track_id]
            self.N = len(self.waypoints)
            self.prv_err = None
            self.cur_err = None
            self.sum_err = 0.0
	    self.i = 0
	    expert_data = np.array([])
	    #lst_L1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 71, 72, 73, 74, 75, 76, 77, 78, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 211, 212, 213, 214, 215, 251, 252, 253, 254, 255, 301, 302, 303, 304, 305, 306, 307, 351, 352, 353, 354, 355, 356, 357, 461, 462, 463, 464, 465, 466, 467, 501, 502, 503, 504, 505]
	    #lst_L2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
	    #lst_L = [151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325]
	    lst_L = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515]
	    #lst_R1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 71, 72, 73, 74, 75, 181, 182, 183, 184, 185, 211, 212, 213, 214, 215, 251, 252, 253, 254, 255, 311, 312, 313, 314, 315, 306, 307, 351, 352, 353, 354, 355, 356, 357, 471, 472, 473, 474, 475, 511, 512, 513, 514, 515]
	    #lst_R2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 51, 52, 53, 54, 55]
	    #lst_R3 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515]
	    lst_R = []

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
                
                cur_pos = self.env.get_pose()
                
                ######################################## YOU CAN ONLY CHANGE THIS PART ########################################
                # TO DO
                """ 
                1) Find nearest waypoint from a race-car
                2) Calculate error between the race-car heading and the direction vector between the lookahead waypoint and the race-car
                3) Determine input steering of the race-car using PID controller
                4) Calculate input velocity of the race-car appropriately in terms of input steering
                """
                
		# 1) Find nearest waypoint from a race-car
		idx_nearest, dist_nearest = 0, 100000
		for i in range(self.N):
		    if dist(self.waypoints[i], cur_pos.position) <= dist_nearest:
			dist_nearest = dist(self.waypoints[i], cur_pos.position)
			idx_nearest = i
		
		# 2) Calculate error between the race-car heading and the direction vector between the lookahead waypoint and the race-car
		if idx_nearest + self.lookahead < self.N:
		    idx_target = idx_nearest + self.lookahead
		else:
		    idx_target = idx_nearest + self.lookahead - self.N
		waypoint_target = self.waypoints[idx_target]
		theta_g = math.atan2(waypoint_target[1] - cur_pos.position.y, waypoint_target[0] - cur_pos.position.x)
		try:
		    pre_pos = self.pre_pos
		    theta_h = math.atan2(cur_pos.position.y - pre_pos.position.y, cur_pos.position.x- pre_pos.position.x)
		except:
		    theta_h = 0
		self.pre_pos = cur_pos
		err = theta_g - theta_h
		while err > math.pi or err < -math.pi:		
		    if err > math.pi:
			err = err - 2 * math.pi
		    else:
			err = err + 2 * math.pi

		self.cur_err = -err

		# 3) Determine input steering of the race-car using PID controller
		if self.prv_err != None:
		    input_steering = self.Kp*self.cur_err + self.Ki*self.dt*self.sum_err + self.Kd/self.dt*(self.cur_err - self.prv_err)
		else:
		    input_steering = self.Kp*self.cur_err + self.Ki*self.dt*self.sum_err + self.Kd/self.dt*self.cur_err
		self.prv_err = self.cur_err
		self.sum_err = self.sum_err + self.cur_err
		    
		if self.i in lst_L:
		    print(self.i, ":, in lst_L")
		    input_steering = input_steering - 1
		elif self.i in lst_R:
		    print(self.i, ":, in lst_R")
		    input_steering = input_steering + 1	
		# 4) Calculate input velocity of the race-car appropriately in terms of input steering
		#input_steering = input_steering + 1*(np.random.rand(1)[0]-0.5)
		if input_steering > 1.5:
		    input_steering = 1.5
		elif input_steering < -1.5:
		    input_steering = -1.5
		input_vel = 1.5 #- abs(self.cur_err)/1.6
                obs, _, done, logs = self.env.step([input_steering, input_vel])
		obs = np.reshape(obs, [1, -1])
		if expert_data.shape == (0, ):
			expert_data = obs
		else:
		    expert_data = np.concatenate((expert_data, obs), axis=0)
		print(self.i, ":", "input_steering:", input_steering)




		self.i = self.i + 1
                ######################################## YOU CAN ONLY CHANGE THIS PART ########################################
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
                    if logs['info'] == 1:
                        rt.fail_type = "Collision"
                    if logs['info'] == 2:
                        rt.fail_type = "Exceed Time Limit"
                    self.rt_pub.publish(rt)
                    break
        except:
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
	
        val_data = np.zeros((expert_data.shape[0]-1, expert_data.shape[1]))
        val_data[:, :-2] = expert_data[:-1, :-2]
        val_data[:, -2:] = expert_data[1:, -2:]	

	for i in reversed(lst_L):
	    if i > 0:
		if i+1 not in lst_L:
		    val_data = np.delete(val_data, i+1, axis=0)
		    val_data = np.delete(val_data, i, axis=0)
		else:
		    val_data = np.delete(val_data, i, axis=0)


	val_data = np.delete(val_data, 0, axis=0)

        #######################################################
        # choose the name of the data
        DATA_PATH = 'data_track3_L.csv'
        #######################################################
        try:
            ex_data = np.loadtxt(project_path + '/project/IS_ROBOLUCK/project/data/' + DATA_PATH, delimiter=",")
        except:
            expert_data = val_data
            np.savetxt(project_path + '/project/IS_ROBOLUCK/project/data/' + DATA_PATH, expert_data, delimiter=",")
        else:
            expert_data = np.concatenate((ex_data, val_data), axis=0)
            np.savetxt(project_path + '/project/IS_ROBOLUCK/project/data/' + DATA_PATH, expert_data, delimiter=",")
        print("Successfully saved the expert data", expert_data.shape)
        if is_exit:
            rospy.signal_shutdown("End query")
        
        return
if __name__ == '__main__':
    with open(yaml_file) as file:
        args = yaml.load(file)
    PurePursuit(args)
    rospy.spin()
