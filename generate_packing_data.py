"""
@file generate_pictures.py
@copyright Software License Agreement (BSD License).
Copyright (c) 2017, Rutgers the State University of New Jersey, New Brunswick.
All Rights Reserved. For a full description see the file named LICENSE.
Authors: Chaitanya Mitash, Kostas Bekris, Abdeslam Boularias.
"""

import os, sys, shutil
import os.path as osp
import time, random
from datetime import datetime
import rospy
from geometry_msgs.msg import Pose

from ConfigParser import ConfigParser
import Label

import numpy

random.seed(datetime.now())

if os.environ.get('BLENDER_PATH') == None:
    print("Please set BLENDER_PATH in bashrc!")
    sys.exit()

# Verify if repository path is set in bashrc
if os.environ.get('PHYSIM_GENDATA') == None:
    print("Please set PHYSIM_GENDATA in bashrc!")
    sys.exit()

g_repo_path = os.environ['PHYSIM_GENDATA']

# Initialization
g_blender_executable_path = os.environ['BLENDER_PATH']

blank_file = osp.join('blank.blend')
empty_bin_file = osp.join('empty_bin.blend')

push_code = osp.join('push.py')
drop_code = osp.join('drop_and_render.py')

cfg = ConfigParser("config.yml", "camera_info.yml")
frame_number = cfg.getNumSimulationSteps() - 1
pLabel = Label.Label()


def get_initial_pose_and_control(pc_path, target_pose, noise):
	# pc_path: Path to the point cloud representing the current state
	# target_pose: Target pose for the object being manipulated
	# noise: Simulating perception/execution noise

	# currently set as a static offset wrt target pose; this is generally computed using the sensing data
	offset_position = [0.03, -0.03, 0.01]

	init_pose = [target_pose.position.x, target_pose.position.y, target_pose.position.z,
				 target_pose.orientation.w, target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z]

	init_pose[0] += offset_position[0]
	init_pose[1] += offset_position[1]
	init_pose[2] += offset_position[2]

	# control vector should be returned by the control algorithm based on sensing data
	target_vector = [target_pose.position.x - init_pose[0], 
					 target_pose.position.y - init_pose[1], 
					 target_pose.position.z - init_pose[2]]

	# noise can be added to this init pose to simulate perception/execution noise

	return init_pose, target_vector

syn_images_folder = 'rendered_images'
if os.path.exists(syn_images_folder):
    shutil.rmtree(syn_images_folder)
os.mkdir(syn_images_folder)
os.mkdir(syn_images_folder + "/debug")

# Render Empty Bin
scene_num = 0
try:
    render_cmd = '%s %s -b --python %s -- %d' % \
    (g_blender_executable_path, blank_file, drop_code, scene_num)

    os.system(render_cmd)
except:
    print('render failed. render_cmd: %s' % (render_cmd))

pLabel.save_pointcloud_data(syn_images_folder, scene_num, 1)

for scene_id in range(1, 10):
	pc_path = os.path.join(g_repo_path, 'rendered_images/', 'image_%05d/depth/pointcloud.pcd' % (scene_id - 1))
	scene_path = os.path.join(g_repo_path, 'object_poses/%05d_01.yml' % scene_id)
	
	target_pose = Pose()
	numObjectsInScene, next_pose = cfg.getObjPoses(scene_path, 1)
	target_pose.position.x = next_pose[0][0]
	target_pose.position.y = next_pose[0][1]
	target_pose.position.z = next_pose[0][2]
	target_pose.orientation.x = next_pose[0][4]
	target_pose.orientation.y = next_pose[0][5]
	target_pose.orientation.z = next_pose[0][6]
	target_pose.orientation.w = next_pose[0][3]

	#add random noise
	noise = []
	
	# initial pose of the object to be manipulated and target vector for the end-effector
	init_pose, target_vector = get_initial_pose_and_control(pc_path, target_pose, noise)

	# push object from the initial configuration along the target vector
	try:
		render_cmd = '%s %s -b --python %s -- %d %f %f %f %f %f %f %f %f %f %f' % \
		(g_blender_executable_path, empty_bin_file, push_code, scene_id,
		 init_pose[0], init_pose[1], init_pose[2], init_pose[3], init_pose[4], init_pose[5], init_pose[6], 
		 target_vector[0], target_vector[1], target_vector[2])

		os.system(render_cmd)
	except:
		print('render failed. render_cmd: %s' % (render_cmd))

	# drop the object once the push is complete
	try:
		render_cmd = '%s %s -b --python %s -- %d' % \
		(g_blender_executable_path, empty_bin_file, drop_code, scene_id)

		os.system(render_cmd)
	except:
		print('render failed. render_cmd: %s' % (render_cmd))

	pLabel.save_pointcloud_data(syn_images_folder, scene_id, frame_number)
	print ("image generation complete!")