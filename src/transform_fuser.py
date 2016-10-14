#!/usr/bin/env python
import sys
import cv2
import math
import numpy as np
import bayesplane
import plane
import transformation as tf
import rospy
from copy import deepcopy
import copy as copy_module
from apriltags.msg import AprilTagDetections
from visualization_msgs.msg import Marker
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def fuse_transform(marker, rgb_image, depth_image, camera_intrinsics, frame_id, ax=None):
	I = np.array(camera_intrinsics).reshape(3,3) # camera intrinsics
	fx = I.item(0,0)
	fy = I.item(1,1)
	px = I.item(0,2)
	py = I.item(1,2)
	detection_id = marker.id
	tag_corners = marker.corners2d
	tag_size = marker.tag_size
	corners0_x = tag_corners[0].x
	corners0_y = tag_corners[0].y
	corners1_x = tag_corners[1].x
	corners1_y = tag_corners[1].y
	corners2_x = tag_corners[2].x 
	corners2_y = tag_corners[2].y
	corners3_x = tag_corners[3].x 
	corners3_y = tag_corners[3].y
	position_x = marker.pose.position.x
	position_y = marker.pose.position.y 
	position_z = marker.pose.position.z
	rotation_x = marker.pose.orientation.x 
	rotation_y = marker.pose.orientation.y 
	rotation_z = marker.pose.orientation.z 
	rotation_w = marker.pose.orientation.w

	x_array = [corners0_x, corners1_x, corners2_x, corners3_x]
	y_array = [corners0_y, corners1_y, corners2_y, corners3_y]
	x_array = sorted(x_array)
	y_array = sorted(y_array)
	x_min = int(x_array[1]) + 0
	x_max = int(x_array[2]) - 0
	y_min = int(y_array[1]) + 0
	y_max = int(y_array[2]) - 0

	# cv2.imshow('image', rgb_image[y_min:y_max, x_min:x_max])
	# cv2.waitKey(0)
	# Sample the depth points

	
	# generate_plot(samples_depth, ax)
	# Sample the rgb points
	M = tf.quaternion_matrix([rotation_w,rotation_x,rotation_y,rotation_z]) 
	M[0, 3] = position_x
	M[1, 3] = position_y
	M[2, 3] = position_z
	M_d = np.delete(M, 3, 0)
	C = np.dot(I, M_d)
	x_samples = np.linspace(-0.01, 0.01, num = 10)
	y_samples = np.linspace(-0.01, 0.01, num = 10)
	sample_rgb = []
	for i in x_samples:
		for j in y_samples:
			sample_rgb.append([i,j,0,1])
	sample_rgb = np.transpose(np.array(sample_rgb))
	sample_rgb = np.transpose(np.dot(M_d, sample_rgb))
	cov = np.asarray([1] * sample_rgb.shape[0])
	rgb_plane_est = bayesplane.fit_plane_bayes(sample_rgb, cov)

	all_pts = []
	for i in range(x_min, x_max):
		for j in range(y_min, y_max):
			depth = depth_image[j,i].item(0) / 1000.0
			if(depth != 0) and (np.average(rgb_image[j,i] > 140)):
				x = (i - px) * depth / fx
				y = (j - py) * depth / fy
				all_pts.append([x,y,depth])
	# print len(all_pts)
	sample_cov = 0.9
	samples_depth = np.array(all_pts)
	cov = np.asarray([sample_cov] * samples_depth.shape[0])
	#print samples_depth.shape
	if(len(samples_depth.shape) > 1):
		depth_plane_est = bayesplane.fit_plane_bayes(samples_depth, cov)
		w,h = np.shape(sample_rgb)
		rgb_center = sample_rgb[w / 2, :]
		w,h = np.shape(samples_depth)
		depth_center = samples_depth[w / 2, :]

		# Generate the rotation
		vector_rgb = rgb_plane_est.mean.vectorize()[0:3]
		vector_depth = depth_plane_est.mean.vectorize()[0:3]
		vector_cross = np.cross(vector_rgb, vector_depth)
		vector_sin = np.linalg.norm(vector_cross)

		vector_cos = np.dot(vector_rgb, vector_depth)
		vector_skew = np.array([[0, -vector_cross[2], vector_cross[1]],
								   [vector_cross[2], 0, -vector_cross[0]],
								   [-vector_cross[1], vector_cross[0], 0]])
		vector_eye = np.eye(3)
		R = vector_eye + vector_skew + np.linalg.matrix_power(vector_skew, 2) * (1 - vector_cos) / (vector_sin * vector_sin)
		angle_error = vector_sin / (np.linalg.norm(vector_rgb) * np.linalg.norm(vector_depth));
		# Find the new normal from the rotation matrix
		rotate_mat = np.eye(4)
		rotate_mat[0:3, 0:3] = R
		sub_center = np.eye(4)
		sub_center[0:3, 3] = -1*rgb_center.T
		add_center = np.eye(4)
		add_center[0:3, 3] = rgb_center.T
		post_rotate = np.dot(add_center, np.dot(rotate_mat, sub_center))
		if angle_error > 0.3:
			M_r = np.dot(post_rotate, M)
		else:
			M_r = M
	else:
		M_r = M 

	R_fused = M_r[0:3, 0:3]
	T_fused = M_r[0:3, 3]
	quart_fused = tf.quaternion_from_matrix(R_fused)
	fused_marker = copy_module.deepcopy(marker)
	fused_marker.pose.position.x = T_fused[0]
	fused_marker.pose.position.y = T_fused[1]
	fused_marker.pose.position.z = T_fused[2]
	fused_marker.pose.orientation.x = quart_fused[1]
	fused_marker.pose.orientation.y = quart_fused[2]
	fused_marker.pose.orientation.z = quart_fused[3]
	fused_marker.pose.orientation.w = quart_fused[0]
	transformed_marker = generate_marker(detection_id, R_fused, T_fused, tag_size, fused_marker, frame_id)
	return fused_marker, transformed_marker

def generate_marker(tag_id, rotation, translation, tag_size, fused_detection, frame_id):
	marker_transform = Marker();
	marker_transform.header.frame_id = frame_id
     # marker_transform.header.frame_id = msg->header.frame_id;
     # marker_transform.header.stamp = msg->header.stamp;

     #    // Only publish marker for 0.5 seconds after it
     #    // was last seen
	marker_transform.lifetime = rospy.Duration.from_sec(0.5)
	marker_transform.id = tag_id;
	marker_transform.type= marker_transform.CUBE
	marker_transform.scale.x = tag_size;
	marker_transform.scale.y = tag_size;
	marker_transform.scale.z = 0.01;
	marker_transform.action = marker_transform.ADD
	marker_transform.pose = fused_detection.pose;
	# marker_transform.pose.position.x = translation[0];
	# marker_transform.pose.position.y = translation[1];
	# marker_transform.pose.position.z = translation[2];
	# marker_transform.pose.orientation.x = ;
	# marker_transform.pose.orientation.y = q.y();
	# marker_transform.pose.orientation.z = q.z();
	# marker_transform.pose.orientation.w = q.w();

	marker_transform.color.r = 0.0;
	marker_transform.color.g = 1.0;
	marker_transform.color.b = 1.0;
	marker_transform.color.a = 1.0;
	return marker_transform

def generate_plot(depth_points, ax):
			# Debugger plot 
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d') 
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax.scatter(depth_points[:, 0], depth_points[:, 1], depth_points[:, 2], c='g')
	plt.show()

