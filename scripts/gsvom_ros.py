#!/usr/bin/env python
import numpy as np
from PIL import Image

import rospy
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import tf2_ros
import tf
import ros_numpy
from cv_bridge import CvBridge

import gsvom
from semantic_association.association_models_factory import get_trained_model


class VoxelMapper:
    def __init__(self):
        # Received data storage variables
        self.robot_position = None
        self.camera1_intrinsics = None
        self.camera1_segmented_image = None
        self.camera1_to_world_matrix = None
        self.camera2_intrinsics = None
        self.camera2_segmented_image = None
        self.camera2_to_world_matrix = None
        self.camera3_intrinsics = None
        self.camera3_segmented_image = None
        self.camera3_to_world_matrix = None

        # Coordinate transformation related variables
        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        self.tf_transformer = tf.TransformerROS()

        # Standard G-VOM parameters
        self.xy_resolution = rospy.get_param("~xy_resolution", 0.15)
        z_resolution = rospy.get_param("~z_resolution", 0.15)
        self.width = rospy.get_param("~width", 256)
        height = rospy.get_param("~height", 64)
        buffer_size = rospy.get_param("~buffer_size", 1)
        min_point_distance = rospy.get_param("~min_point_distance", 1.0)
        positive_obstacle_threshold = rospy.get_param("~positive_obstacle_threshold", 0.50)
        negative_obstacle_threshold = rospy.get_param("~negative_obstacle_threshold", 0.5)
        slope_obsacle_threshold = rospy.get_param("~slope_obsacle_threshold", 0.3)
        robot_height = rospy.get_param("~robot_height", 1.0)
        robot_radius = rospy.get_param("~robot_radius", 0.75)
        ground_to_lidar_height = rospy.get_param("~ground_to_lidar_height", 0.6)
        xy_eigen_dist = rospy.get_param("~xy_eigen_dist", 1)
        z_eigen_dist = rospy.get_param("~z_eigen_dist", 1)
        use_dynamic_combined_map = rospy.get_param("~use_dynamic_combined_map", True)
        # Semantics parameters
        semantic_label_length = 1
        number_of_semantic_labels = 52
        semantic_assignment_distance = 128
        geometric_context_size = 9
        model_type = rospy.get_param("~association_model_type")
        model_weights_path = rospy.get_param("~association_model_weights_path")
        geometric_feature_type = rospy.get_param("~geometric_feature_type")
        feature_extractor_weights_path = rospy.get_param("~feature_extractor_weights_path")
        # Postprocessing parameters
        self.density_threshold = rospy.get_param("~density_threshold", 50)
        self.min_roughness = rospy.get_param("~min_roughness", -10)
        self.max_roughness = rospy.get_param("~max_roughness", 0)
        # Auxiliary parameters
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        map_merging_frequency = rospy.get_param("~map_freq", 10.0)  # Hz
        semantics_merging_frequency = rospy.get_param("~semantics_freq", 5.0) # Hz
        visualization_colors_file_path = rospy.get_param("~visualization_colors_file")

        # Prepare the semantics to voxels association method
        association_model, feature_extractor, place_label_threshold, skip_pixels = get_trained_model(model_type, number_of_semantic_labels, model_weights_path,
                                                                                        geometric_feature_type, feature_extractor_weights_path)

        # Prepare G-SVOM itself
        self.voxel_mapper = gsvom.Gsvom(self.xy_resolution,
                                        z_resolution,
                                        self.width,
                                        height,
                                        buffer_size,
                                        min_point_distance,
                                        positive_obstacle_threshold,
                                        negative_obstacle_threshold,
                                        slope_obsacle_threshold,
                                        robot_height,
                                        robot_radius,
                                        ground_to_lidar_height,
                                        xy_eigen_dist,
                                        z_eigen_dist,
                                        semantic_label_length,
                                        number_of_semantic_labels,
                                        semantic_assignment_distance,
                                        geometric_context_size,
                                        association_model,
                                        feature_extractor,
                                        place_label_threshold,
                                        skip_pixels,
                                        use_dynamic_combined_map)

        # Image processing and visualization variables
        self.class_colors = np.loadtxt(visualization_colors_file_path, dtype=np.float32)
        if self.class_colors.shape[0] != number_of_semantic_labels:
            rospy.logerr("[G-SVOM] The number of label visualization colors doesn't match the number of semantic labels!")
            return
        self.class_colors /= 255
        self.ros_cv_bridge = CvBridge()

        # Input data subscribers
        self.sub_cloud = rospy.Subscriber("~cloud", PointCloud2, self.cb_lidar, queue_size=1)
        self.sub_odom = rospy.Subscriber("~odom", Odometry, self.cb_odom, queue_size=1)
        self.sub_image1 = rospy.Subscriber("~segmented_image1", Image, self.cb_image1, queue_size=1)
        self.sub_camera_info1 = rospy.Subscriber("~camera_info1", CameraInfo, self.cb_camera1_info, queue_size=1)
        self.sub_image2 = rospy.Subscriber("~segmented_image2", Image, self.cb_image2, queue_size=1)
        self.sub_camera_info2 = rospy.Subscriber("~camera_info2", CameraInfo, self.cb_camera2_info, queue_size=1)
        self.sub_image3 = rospy.Subscriber("~segmented_image3", Image, self.cb_image3, queue_size=1)
        self.sub_camera_info3 = rospy.Subscriber("~camera_info3", CameraInfo, self.cb_camera3_info, queue_size=1)

        # Output data publishers
        self.s_obstacle_map_pub = rospy.Publisher("~soft_obstacle_map", OccupancyGrid, queue_size=1)
        self.n_obstacle_map_pub = rospy.Publisher("~negative_obstacle_map", OccupancyGrid, queue_size=1)
        self.h_obstacle_map_pub = rospy.Publisher("~hard_obstacle_map", OccupancyGrid, queue_size=1)
        self.g_certainty_pub = rospy.Publisher("~ground_certainty_map", OccupancyGrid, queue_size=1)
        self.r_map_pub = rospy.Publisher("~roughness_map", OccupancyGrid, queue_size=1)

        # Debug data publishers
        self.voxel_hm_debug_pub = rospy.Publisher('~debug/height_map', PointCloud2, queue_size=1)
        self.voxel_inf_hm_debug_pub = rospy.Publisher('~debug/inferred_height_map', PointCloud2, queue_size=1)
        self.colored_map_debug_pub = rospy.Publisher('~debug/colored_pointcloud', PointCloud2, queue_size=1)

        # Map merging and semantics association timers
        self.map_merge_timer = rospy.Timer(rospy.Duration(1.0/map_merging_frequency), self.cb_map_merge_timer)
        self.semantics_merge_timer = rospy.Timer(rospy.Duration(1.0/semantics_merging_frequency), self.cb_merge_semantics)

        rospy.loginfo("[G-SVOM] Voxel mapper successfully started!")

    def cb_odom(self, data):
        self.robot_position = (data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z)

    def cb_lidar(self, data):
        if self.robot_position is None:
            rospy.logwarn("[G-SVOM] No robot position recorded!")
            return

        robot_pos = self.robot_position
        lidar_to_world_transform = self.get_transform_as_matrix(self.odom_frame, data.header.frame_id, data.header.stamp)
        point_cloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)
        self.voxel_mapper.process_pointcloud(point_cloud, robot_pos, lidar_to_world_transform)

    def cb_map_merge_timer(self, event):
        map_data = self.voxel_mapper.combine_maps()
        if map_data is None:
            rospy.logwarn("[G-SVOM] No map data to publish!")
            return

        map_origin = map_data[0]
        positive_obstacle_map = map_data[1]
        negative_obstacle_map = map_data[2]
        roughness_map = map_data[3]
        cert_map = map_data[4]

        current_time = rospy.Time.now()

        out_map = OccupancyGrid()
        out_map.header.stamp = current_time
        out_map.header.frame_id = self.odom_frame
        out_map.info.resolution = self.xy_resolution
        out_map.info.width = self.width
        out_map.info.height = self.width
        out_map.info.origin.orientation.x = 0
        out_map.info.origin.orientation.y = 0
        out_map.info.origin.orientation.z = 0
        out_map.info.origin.orientation.w = 1
        out_map.info.origin.position.x = map_origin[0]
        out_map.info.origin.position.y = map_origin[1]
        out_map.info.origin.position.z = 0

        # Hard obstacles
        out_map.data = np.reshape(np.maximum(100 * (positive_obstacle_map > self.density_threshold), negative_obstacle_map), -1, order='F').astype(np.int8)
        self.h_obstacle_map_pub.publish(out_map)

        # Soft obstacles
        out_map.data = np.reshape(100 * (positive_obstacle_map <= self.density_threshold) * (positive_obstacle_map > 0), -1, order='F').astype(np.int8)
        self.s_obstacle_map_pub.publish(out_map)

        # Ground certainty
        out_map.data = np.reshape(cert_map * 100, -1, order='F').astype(np.int8)
        self.g_certainty_pub.publish(out_map)

        # Negative obstacles
        out_map.data = np.reshape(negative_obstacle_map, -1, order='F').astype(np.int8)
        self.n_obstacle_map_pub.publish(out_map)

        # Roughness
        roughness_range = self.max_roughness - self.min_roughness
        roughness_map = 100 * ((np.maximum(np.minimum(roughness_map, self.max_roughness), self.min_roughness) + self.min_roughness) / roughness_range)
        out_map.data = np.reshape(roughness_map, -1, order='F').astype(np.int8)
        self.r_map_pub.publish(out_map)

        ###### Debug maps ######
        # Voxel height map
        voxel_hm = self.voxel_mapper.make_debug_height_map()
        if voxel_hm is not None:
            field_values = [voxel_hm[:, 0], voxel_hm[:, 1], voxel_hm[:, 2], voxel_hm[:, 3], voxel_hm[:, 4], voxel_hm[:, 5], voxel_hm[:, 6],
                            positive_obstacle_map.flatten('F')]
            field_names = 'x,y,z,roughness,slope_x,slope_y,slope,obstacles'
            voxel_hm = np.core.records.fromarrays(field_values, names=field_names)
            self.voxel_hm_debug_pub.publish(ros_numpy.point_cloud2.array_to_pointcloud2(voxel_hm, current_time, self.odom_frame))

        # Inferred height map
        voxel_inf_hm = self.voxel_mapper.make_debug_inferred_height_map()
        if voxel_inf_hm is not None:
            field_values = [voxel_inf_hm[:, 0], voxel_inf_hm[:, 1], voxel_inf_hm[:, 2]]
            field_names = 'x,y,z'
            voxel_inf_hm = np.core.records.fromarrays(field_values, names=field_names)
            self.voxel_inf_hm_debug_pub.publish(ros_numpy.point_cloud2.array_to_pointcloud2(voxel_inf_hm, current_time, self.odom_frame))

        rospy.loginfo("[G-SVOM] Published maps!")

    def cb_camera1_info(self, data):
        self.camera1_intrinsics = data.K

    def cb_image1(self, data):
        cv_image = self.ros_cv_bridge.imgmsg_to_cv2(data, desired_encoding="mono8")
        self.camera1_segmented_image = np.expand_dims(cv_image.T, axis=-1)
        self.camera1_to_world_matrix = self.get_transform_as_matrix(self.odom_frame, data.header.frame_id, data.header.stamp)

    def cb_camera2_info(self, data):
        self.camera2_intrinsics = data.K

    def cb_image2(self, data):
        cv_image = self.ros_cv_bridge.imgmsg_to_cv2(data, desired_encoding="mono8")
        self.camera2_segmented_image = np.expand_dims(cv_image.T, axis=-1)
        self.camera2_to_world_matrix = self.get_transform_as_matrix(self.odom_frame, data.header.frame_id, data.header.stamp)

    def cb_camera3_info(self, data):
        self.camera3_intrinsics = data.K

    def cb_image3(self, data):
        cv_image = self.ros_cv_bridge.imgmsg_to_cv2(data, desired_encoding="mono8")
        self.camera3_segmented_image = np.expand_dims(cv_image.T, axis=-1)
        self.camera3_to_world_matrix = self.get_transform_as_matrix(self.odom_frame, data.header.frame_id, data.header.stamp)

    def cb_merge_semantics(self, event):
        # Merge semantics
        merged_semantics = False
        if not (self.camera1_intrinsics is None or self.camera1_segmented_image is None or self.camera1_to_world_matrix is None):
            intrinsic_matrix = np.array(self.camera1_intrinsics).reshape((3, 3))
            self.voxel_mapper.process_semantics(self.camera1_segmented_image.astype(np.int64), intrinsic_matrix, self.camera1_to_world_matrix)
            merged_semantics = True
        if not (self.camera2_intrinsics is None or self.camera2_segmented_image is None or self.camera2_to_world_matrix is None):
            intrinsic_matrix = np.array(self.camera2_intrinsics).reshape((3, 3))
            self.voxel_mapper.process_semantics(self.camera2_segmented_image.astype(np.int64), intrinsic_matrix, self.camera2_to_world_matrix)
            merged_semantics = True
        if not (self.camera3_intrinsics is None or self.camera3_segmented_image is None or self.camera3_to_world_matrix is None):
            intrinsic_matrix = np.array(self.camera3_intrinsics).reshape((3, 3))
            self.voxel_mapper.process_semantics(self.camera3_segmented_image.astype(np.int64), intrinsic_matrix, self.camera3_to_world_matrix)
            merged_semantics = True
        if not merged_semantics:
            rospy.logwarn("[G-SVOM] Didn't merge any semantics!")

        # Publish painted occupancy pointcloud
        vis_data = self.voxel_mapper.get_map_as_painted_occupancy_pointcloud()
        if vis_data is None:
            rospy.logwarn("[G-SVOM] No painted point cloud to show!")
            return

        voxel_centers, voxel_labels = vis_data
        voxel_labels = voxel_labels.squeeze().astype(int)
        voxel_colors = self.class_colors[voxel_labels]

        field_data = [voxel_centers[:, 0], voxel_centers[:, 1], voxel_centers[:, 2], voxel_colors[:, 0], voxel_colors[:, 1], voxel_colors[:, 2]]
        field_names = "x,y,z,r,g,b"
        publish_data = np.core.records.fromarrays(field_data, names=field_names)
        self.colored_map_debug_pub.publish(ros_numpy.point_cloud2.array_to_pointcloud2(publish_data, rospy.Time.now(), self.odom_frame))

        rospy.loginfo("[G-SVOM] Merged semantics!")

    def get_transform_as_matrix(self, target_frame: str, source_frame: str, timestamp):
        try:
            transform = self.tfBuffer.lookup_transform(target_frame, source_frame, timestamp, rospy.Duration(1))
        except tf.ExtrapolationException:
            rospy.logerr(f"[G-SVOM] Failed to get the transform from: '{source_frame}' to '{target_frame}'!")
            return

        translation = np.zeros([3])
        translation[0] = transform.transform.translation.x
        translation[1] = transform.transform.translation.y
        translation[2] = transform.transform.translation.z

        rotation = np.zeros([4])
        rotation[0] = transform.transform.rotation.x
        rotation[1] = transform.transform.rotation.y
        rotation[2] = transform.transform.rotation.z
        rotation[3] = transform.transform.rotation.w

        return self.tf_transformer.fromTranslationRotation(translation, rotation)

            
if __name__ == '__main__':
    rospy.init_node('gsvom_voxel_mapping')
    node = VoxelMapper()

    while not rospy.is_shutdown():
        rospy.spin()

    rospy.on_shutdown(node.on_shutdown)
