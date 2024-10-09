#!/usr/bin/env python
import numpy as np
import cv2
from PIL import Image
import io
import requests
import base64

import rospy
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import PointCloud2, CompressedImage
import tf2_ros
import tf
import ros_numpy

import gsvom
from semantic_association.association_models_factory import get_trained_model


class VoxelMapper:
    def __init__(self):
        self.robot_position = None

        # TODO: Update this to Nones to be safe from doing unexpected things
        self.intrinsic_matrix = np.array([[0.0, 1.0, 10.0],
                                          [0.0, 1.0, 10.0],
                                          [0.0, 0.0, 1.0]])
        self.distortion_parameters = [0.0, 0.0, 0.0, 0.0]
        self.camera_frame = "/hdr_front"
        self.camera_to_world_transform = None


        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        self.tf_transformer = tf.TransformerROS()

        # Standard G-VOM parameters
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.xy_resolution = rospy.get_param("~xy_resolution", 0.15)
        self.z_resolution = rospy.get_param("~z_resolution", 0.15)
        self.width = rospy.get_param("~width", 256)
        self.height = rospy.get_param("~height", 64)
        self.buffer_size = rospy.get_param("~buffer_size", 1)
        self.min_point_distance = rospy.get_param("~min_point_distance", 1.0)
        self.positive_obstacle_threshold = rospy.get_param("~positive_obstacle_threshold", 0.50)
        self.negative_obstacle_threshold = rospy.get_param("~negative_obstacle_threshold", 0.5)
        self.density_threshold = rospy.get_param("~density_threshold", 50)
        self.slope_obsacle_threshold = rospy.get_param("~slope_obsacle_threshold", 0.3)
        self.min_roughness = rospy.get_param("~min_roughness", -10)
        self.max_roughness = rospy.get_param("~max_roughness", 0)
        self.robot_height = rospy.get_param("~robot_height", 1.0)
        self.robot_radius = rospy.get_param("~robot_radius", 0.75)
        self.ground_to_lidar_height = rospy.get_param("~ground_to_lidar_height", 0.6)
        self.freq = rospy.get_param("~freq", 10.0) # Hz
        self.xy_eigen_dist = rospy.get_param("~xy_eigen_dist", 1)
        self.z_eigen_dist = rospy.get_param("~z_eigen_dist", 1)
        # Semantics parameters
        semantic_label_length = 1
        number_of_semantic_labels = 52
        semantic_assignment_distance = 128
        geometric_context_size = 9
        use_dynamic_combined_map = True

        model_type = rospy.get_param("~association_model_type")
        model_weights_path = rospy.get_param("~association_model_weights_path")
        geometric_feature_type = rospy.get_param("~geometric_feature_type")
        feature_extractor_weights_path = rospy.get_param("~feature_extractor_weights_path")
        association_model, feature_extractor, place_label_threshold = get_trained_model(model_type, number_of_semantic_labels, model_weights_path,
                                                                                        geometric_feature_type, feature_extractor_weights_path)
        
        self.voxel_mapper = gsvom.Gsvom(
            self.xy_resolution,
            self.z_resolution,
            self.width,
            self.height,
            self.buffer_size,
            self.min_point_distance,
            self.positive_obstacle_threshold,
            self.negative_obstacle_threshold,
            self.slope_obsacle_threshold,
            self.robot_height,
            self.robot_radius,
            self.ground_to_lidar_height,
            self.xy_eigen_dist,
            self.z_eigen_dist,
            semantic_label_length,
            number_of_semantic_labels,
            semantic_assignment_distance,
            geometric_context_size,
            association_model,
            feature_extractor,
            place_label_threshold,
            use_dynamic_combined_map)

        self.segmentation_server_address = rospy.get_param("~segmentation_server")
        self.segmentation_classes = ['unlabeled', 'bicycle', 'car', 'traffic light', 'street sign', 'bench', 'umbrella', 'skateboard', 'plate', 'bowl', 'sandwich', 'chair', 'potted plant', 'window', 'door', 'tv', 'remote', 'vase', 'banner', 'bush', 'cardboard', 'ceiling-other', 'cloth', 'cupboard', 'curtain', 'dirt', 'fence', 'floor-other', 'furniture-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'plant-other', 'plastic', 'platform', 'railroad', 'rock', 'roof', 'sea', 'shelf', 'sky-other', 'skyscraper', 'stairs', 'straw', 'structural-other', 'table', 'tree', 'wall-other', 'wood']
        self.segmentation_classes_str = ",".join(self.segmentation_classes[1:])
        self.image_scaling_factor = 0.2

        self.sub_cloud = rospy.Subscriber("~cloud", PointCloud2, self.cb_lidar, queue_size=1)
        self.sub_odom = rospy.Subscriber("~odom", Odometry, self.cb_odom, queue_size=1)
        self.sub_image = rospy.Subscriber("~image", CompressedImage, self.cb_image, queue_size=1)
        
        self.s_obstacle_map_pub = rospy.Publisher("~soft_obstacle_map", OccupancyGrid, queue_size=1)
        self.p_obstacle_map_pub = rospy.Publisher("~positive_obstacle_map", OccupancyGrid, queue_size=1)
        self.n_obstacle_map_pub = rospy.Publisher("~negative_obstacle_map", OccupancyGrid, queue_size=1)
        self.h_obstacle_map_pub = rospy.Publisher("~hard_obstacle_map", OccupancyGrid, queue_size=1)
        self.g_certainty_pub = rospy.Publisher("~ground_certainty_map", OccupancyGrid, queue_size=1)
        self.a_certainty_pub = rospy.Publisher("~all_ground_certainty_map", OccupancyGrid, queue_size=1)
        self.r_map_pub = rospy.Publisher("~roughness_map", OccupancyGrid, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1./self.freq), self.cb_timer)
        
        self.lidar_debug_pub = rospy.Publisher('~debug/lidar', PointCloud2, queue_size=1)
        self.voxel_debug_pub = rospy.Publisher('~debug/voxel', PointCloud2, queue_size=1)
        self.voxel_hm_debug_pub = rospy.Publisher('~debug/height_map', PointCloud2, queue_size=1)
        self.voxel_inf_hm_debug_pub = rospy.Publisher('~debug/inferred_height_map', PointCloud2, queue_size=1)

    def cb_odom(self, data):
        self.robot_position = (data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z)

    def cb_lidar(self, data):
        if self.robot_position is None:
            print("no odom")
            return

        robot_pos = self.robot_position
        lidar_frame = data.header.frame_id
        trans = self.tfBuffer.lookup_transform(self.odom_frame, lidar_frame, data.header.stamp, rospy.Duration(1))

        translation = np.zeros([3])
        translation[0] = trans.transform.translation.x
        translation[1] = trans.transform.translation.y
        translation[2] = trans.transform.translation.z

        rotation = np.zeros([4])
        rotation[0] = trans.transform.rotation.x
        rotation[1] = trans.transform.rotation.y
        rotation[2] = trans.transform.rotation.z
        rotation[3] = trans.transform.rotation.w

        tf_matrix = self.tf_transformer.fromTranslationRotation(translation, rotation)
        pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)
        self.voxel_mapper.process_pointcloud(pc, robot_pos, tf_matrix, 0)

    def cb_image(self, data):
        data_array = np.frombuffer(data.data, dtype=np.uint8)
        cv_image = cv2.imdecode(data_array, cv2.IMREAD_COLOR)
        cv_image = cv2.resize(cv_image, (0, 0), fx=self.image_scaling_factor, fy=self.image_scaling_factor)
        scaled_intrinsic_matrix = self.image_scaling_factor*self.intrinsic_matrix
        scaled_intrinsic_matrix[2, 2] = 1.0

        segmented_image = self.segment_image(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        cv2.imshow("Received image", cv_image)
        cv2.waitKey(50)
        rospy.loginfo(f"Fofof")

    def segment_image(self, image_array):
        image = Image.fromarray(image_array, mode="RGB")
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)

        image_files = {"image": ('image.png', img_io, 'image/png')}
        other_data = {'return_all_categories': 'False', "vocab": self.segmentation_classes_str}

        ret_val = requests.post(self.segmentation_server_address, files=image_files, data=other_data)
        if ret_val.status_code == 200:
            ret_val = ret_val.json()
            shape = ret_val["shape"][1:]
            decoded_bytes = base64.b64decode(ret_val["sem_seg"])
            seg_im = np.frombuffer(decoded_bytes, dtype=int).reshape(shape)

            output_labels = ret_val['vocabulary']
            N = len(output_labels)
            filtered_label_map = np.zeros(N, dtype=int)
            for idx, label in enumerate(output_labels):
                filtered_label_map[idx] = self.segmentation_classes.index(label)
            filtered_label_seg = np.zeros_like(seg_im)
            filtered_label_seg[seg_im != N] = filtered_label_map[seg_im[seg_im != N]]

            return filtered_label_seg
        else:
            rospy.logerr(f"[G-SVOM] Semantic segmentation request failed with code {ret_val.status_code} and message '{ret_val.reason}'")
            return None

    def cb_timer(self, event):
        map_data = self.voxel_mapper.combine_maps()
        if map_data is None:
            rospy.loginfo("map_data is None. returning.")
            return

        map_origin = map_data[0]
        obs_map = map_data[1]
        neg_map = map_data[2]
        rough_map = map_data[3]
        cert_map = map_data[4]

        out_map = OccupancyGrid()
        out_map.header.stamp = rospy.Time.now()
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
        out_map.data = np.reshape(np.maximum(100 * (obs_map > self.density_threshold), neg_map), -1, order='F').astype(np.int8)
        self.h_obstacle_map_pub.publish(out_map)

        # Soft obstacles
        out_map.data = np.reshape(100 * (obs_map <= self.density_threshold) * (obs_map > 0), -1, order='F').astype(np.int8)
        self.s_obstacle_map_pub.publish(out_map)

        # Ground certainty
        out_map.data = np.reshape(cert_map*100, -1, order='F').astype(np.int8)
        self.g_certainty_pub.publish(out_map)
        self.a_certainty_pub.publish(out_map)

        # Negative obstacles
        out_map.data = np.reshape(neg_map, -1, order='F').astype(np.int8)
        self.n_obstacle_map_pub.publish(out_map)

        # Roughness
        rough_map = ((np.maximum(np.minimum(rough_map, self.max_roughness), self.min_roughness) + self.min_roughness) / (self.max_roughness - self.min_roughness)) * 100
        out_map.data = np.reshape(rough_map, -1, order='F').astype(np.int8)
        self.r_map_pub.publish(out_map)

        ###### Debug maps ######
        # Voxel map
        voxel_pc = self.voxel_mapper.make_debug_voxel_map()
        if voxel_pc is not None:
            voxel_pc = np.core.records.fromarrays([voxel_pc[:,0], voxel_pc[:,1], voxel_pc[:,2], voxel_pc[:,3], voxel_pc[:,4], voxel_pc[:,5], voxel_pc[:,6], voxel_pc[:,7]],
                                                  names='x,y,z,solid factor,count,eigen_line,eigen_surface,eigen_point')
            self.voxel_debug_pub.publish(ros_numpy.point_cloud2.array_to_pointcloud2(voxel_pc, rospy.Time.now(), self.odom_frame))

        # Voxel height map
        voxel_hm = self.voxel_mapper.make_debug_height_map()
        if voxel_hm is not None:
            voxel_hm = np.core.records.fromarrays([voxel_hm[:,0], voxel_hm[:,1], voxel_hm[:,2], voxel_hm[:,3], voxel_hm[:,4], voxel_hm[:,5], voxel_hm[:,6], obs_map.flatten('F')],
                                                  names='x,y,z,roughness,slope_x,slope_y,slope,obstacles')
            self.voxel_hm_debug_pub.publish(ros_numpy.point_cloud2.array_to_pointcloud2(voxel_hm, rospy.Time.now(), self.odom_frame))
    
        # Inferred height map
        voxel_inf_hm = self.voxel_mapper.make_debug_inferred_height_map()
        if voxel_inf_hm is not None:
            voxel_inf_hm = np.core.records.fromarrays([voxel_inf_hm[:,0], voxel_inf_hm[:,1], voxel_inf_hm[:,2]], names='x,y,z')
            self.voxel_inf_hm_debug_pub.publish(ros_numpy.point_cloud2.array_to_pointcloud2(voxel_inf_hm, rospy.Time.now(), self.odom_frame))
        rospy.loginfo("[G-SVOM] Published maps!")
            
if __name__ == '__main__':
    rospy.init_node('gsvom_voxel_mapping')
    node = VoxelMapper()
    while not rospy.is_shutdown():
        rospy.spin()

    rospy.on_shutdown(node.on_shutdown)
