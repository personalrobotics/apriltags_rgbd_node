import sys
import rospy
import cv2
import transform_fuser as fuser
import matplotlib.pyplot as plt
from apriltags.msg import AprilTagDetections
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import MarkerArray 
from cv_bridge import CvBridge, CvBridgeError
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer

class apriltag_rgbd:
	def __init__(self):
		# Subscribers 
		# self.marker_array_subscribe = rospy.Subscriber("/apriltags_kinect2/detections", AprilTagDetections, self.detection_callback)
		# self.rgb_image_subscribe = rospy.Subscriber("/head/kinect2/qhd/image_color_rect", Image, self.rgb_callback)
		# self.depth_image_subscribe = rospy.Subscriber("/head/kinect2/qhd/image_depth_rect", Image, self.depth_callback)
		self.camera_info = rospy.Subscriber("/head/kinect2/qhd/camera_info", CameraInfo, self.camera_callback)
		tss = ApproximateTimeSynchronizer([Subscriber("/head/kinect2/qhd/image_color_rect", Image),
							   Subscriber("/head/kinect2/qhd/image_depth_rect", Image), 
							   Subscriber("/apriltags_kinect2/detections", AprilTagDetections)], 1,0.5)
		tss.registerCallback(self.processtag_callback)
		# Vars
		self.camera_intrinsics = None
		self.rgb_image = None
		self.depth_image = None
		self.mark_array = None

		# Converters
		self.bridge = CvBridge()

		# Publisher
		self.marker_publish = rospy.Publisher("/apriltags_kinect2/marker_array_fused", MarkerArray)
		self.detection_publish = rospy.Publisher("/apriltags_kinect2/detections_fused", AprilTagDetections)
		
		# Debugger plot 
		# self.fig = plt.figure()
		# self.ax = self.fig.add_subplot(111, projection='3d') 

	def camera_callback(self, data):
		self.camera_intrinsics = data.K

	def processtag_callback(self, rgb_data, depth_data, tag_data):
		try:
			self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
			self.depth_image = self.bridge.imgmsg_to_cv2(depth_data, "16UC1")
		except CvBridgeError as e:
			print(e)

		all_detections = tag_data.detections
		detections_transformed = []
		marker_transformed = []
		if(self.rgb_image != None and self.depth_image != None):
			for current_detection in all_detections:
				current_fused, current_marker = fuser.fuse_transform(current_detection, self.rgb_image, self.depth_image, self.camera_intrinsics, tag_data.header.frame_id)
				current_marker.ns = 'tag'+str(current_detection.id)
				current_marker.id = current_detection.id
				detections_transformed.append(current_fused)
				marker_transformed.append(current_marker)
			detection_msg = AprilTagDetections()
			detection_msg.detections = detections_transformed
			marker_msg = MarkerArray()
			marker_msg.markers = marker_transformed
			self.detection_publish.publish(detection_msg)
			self.marker_publish.publish(marker_msg)

	# def rgb_callback(self, data):
	# 	try:
	# 		self.rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
	# 	except CvBridgeError as e:
	# 		print(e)

	# def depth_callback(self, data):
	# 	try:
	# 		self.depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
	# 	except CvBridgeError as e:
	# 		print(e)

	def detection_callback(self, data):
		all_detections = data.detections
		detections_transformed = []
		marker_transformed = []
		if(self.rgb_image != None and self.depth_image != None):
			for current_detection in all_detections:
				current_fused, current_marker = fuser.fuse_transform(current_detection, self.rgb_image, self.depth_image, self.camera_intrinsics, data.header.frame_id)
				detections_transformed.append(current_fused)
				marker_transformed.append(current_marker)
			detection_msg = AprilTagDetections()
			detection_msg.detections = detections_transformed
			marker_msg = MarkerArray()
			marker_msg.markers = marker_transformed
			self.detection_publish.publish(detection_msg)
			self.marker_publish.publish(marker_msg)



def main(args):
	apriltag_transformer = apriltag_rgbd()
	rospy.init_node('', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")

if __name__ == '__main__':
	main(sys.argv)
