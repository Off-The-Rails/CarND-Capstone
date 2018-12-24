#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
from keras import backend as K
from keras.models import load_model
import numpy as np

STATE_COUNT_THRESHOLD = 4
SITE_STATE_COUNT_TRESHOLD = 1 
NO_WP = -1
SKIP_VAL = 0


##################################################

SMOOTH = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



##################################################


class TLDetector(object):
    def __init__(self):
        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''

        rospy.init_node('tl_detector')
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.waypoint_ktree = None
        self.waypoints_2d = None
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = NO_WP
        self.state_count = 0
        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(self.config['is_site'])
        self.listener = tf.TransformListener()
        self.skipper = 1
        self.using_classifier = False
        self.light_dict = { TrafficLight.RED: "RED",
                            TrafficLight.UNKNOWN: "OTHER"}

        #########################################################
        ########## ON SITE DETECTOR + CLASSIFIER SETUP ##########
        #########################################################
        
        if self.config['is_site']:

            model = load_model(self.config['tl']['tl_classification_model'])
            resize_width = self.config['tl']['classifier_resize_width']
            resize_height = self.config['tl']['classifier_resize_height']
            self.light_classifier.setup_classifier(model, resize_width, resize_height)
            self.invalid_class_number = 3

            #Detector setup
            self.detector_model = load_model(self.config['tl']['tl_detection_model'], custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef })
            self.detector_model._make_predict_function()
            self.resize_width = self.config['tl']['detector_resize_width']
            self.resize_height = self.config['tl']['detector_resize_height']
            
            self.resize_height_ratio = self.config['camera_info']['image_height']/float(self.resize_height)
            self.resize_width_ratio = self.config['camera_info']['image_width']/float(self.resize_width)
            self.middle_col = self.resize_width/2
            self.is_carla = self.config['tl']['is_carla']
            self.projection_threshold = self.config['tl']['projection_threshold']
            self.projection_min = self.config['tl']['projection_min']
            self.color_mode = self.config['tl']['color_mode']

        ##################################################

        # setup publisher
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        
        # Setup subscriptions
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb)

        
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, lane):
        self.waypoints = lane
        if not self.waypoints_2d:
            self.waypoints_2d = [ [ waypoint.pose.pose.position.x, waypoint.pose.pose.position.y ] for waypoint in lane.waypoints ]
            self.waypoint_ktree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        state_th = STATE_COUNT_THRESHOLD
        if self.config['is_site']:
            state_th = SITE_STATE_COUNT_TRESHOLD

        self.using_classifier = True
        if SKIP_VAL > 1:
            if (self.skipper % SKIP_VAL) != 0:
                self.skipper += 1
                return
            self.skipper = 1
        self.camera_image = msg
        light_wp, light_state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != light_state:
            self.state_count = 0
            self.state = light_state
        elif self.state_count >= state_th:
            self.last_state = self.state
            light_wp = light_wp if light_state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1
        rospy.loginfo("light_state = %s" % self.light_dict[light_state])

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        if self.waypoint_ktree == None:
            return 0
            
        return self.waypoint_ktree.query([x,y],1)[1]

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """


        if self.config['is_site']:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, self.color_mode)
            tl_image = self.detect_traffic_light(cv_image)
            if tl_image is not None:
                state = self.light_classifier.get_site_classification(tl_image)
                state = state if (state != self.invalid_class_number) else TrafficLight.UNKNOWN
                rospy.loginfo("[TL_DETECTOR] Nearest TL-state is: %s", state)
                rospy.loginfo("[TL_DETECTOR] RED TL-state is: %s", TrafficLight.RED)
                rospy.loginfo("[TL_DETECTOR] YELLOW TL-state is: %s", TrafficLight.YELLOW)
                rospy.loginfo("[TL_DETECTOR] GREEN TL-state is: %s", TrafficLight.GREEN)
        else:
            # use rgb8 instead of bgr8 for detect
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
            #Get classification
            light_state = self.light_classifier.get_classification(cv_image)

        return light_state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        light_wp = NO_WP

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if self.pose and self.waypoints:
            car_position = self.get_closest_waypoint(self.pose.pose.position.x,self.pose.pose.position.y)
        # find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0],line[1])
                d = temp_wp_idx - car_position
                if (d >= 0 and d < diff) or self.config['is_site']:
                    diff = d
                    closest_light = light
                    light_wp = temp_wp_idx

        if closest_light != None:
            state = self.get_light_state(closest_light)
            return light_wp, state
        self.waypoints = None
        return light_wp, TrafficLight.UNKNOWN

    def detect_traffic_light(self, cv_image):
        resize_image = cv2.cvtColor(cv2.resize(cv_image, (self.resize_width, self.resize_height)), cv2.COLOR_RGB2GRAY)
        resize_image = resize_image[..., np.newaxis]
        if self.is_carla:
            mean = np.mean(resize_image) # mean for data centering
            std = np.std(resize_image) # std for data normalization

            resize_image -= mean
            resize_image /= std

        image_mask = self.detector_model.predict(resize_image[None, :, :, :], batch_size=1)[0]
        image_mask = (image_mask[:,:,0]*255).astype(np.uint8)
        return self._extract_image(image_mask, cv_image)

    def _extract_image(self, pred_image_mask, image):
        if (np.max(pred_image_mask) < self.projection_min):
            return None

        row_projection = np.sum(pred_image_mask, axis = 1)
        row_index =  np.argmax(row_projection)

        if (np.max(row_projection) < self.projection_threshold):
            return None

        zero_row_indexes = np.argwhere(row_projection <= self.projection_threshold)
        top_part = zero_row_indexes[zero_row_indexes < row_index]
        top = np.max(top_part) if top_part.size > 0 else 0
        bottom_part = zero_row_indexes[zero_row_indexes > row_index]
        bottom = np.min(bottom_part) if bottom_part.size > 0 else self.resize_height

        roi = pred_image_mask[top:bottom,:]
        column_projection = np.sum(roi, axis = 0)

        if (np.max(column_projection) < self.projection_min):
            return None

        non_zero_column_index = np.argwhere(column_projection > self.projection_min)

        index_of_column_index =  np.argmin(np.abs(non_zero_column_index - self.middle_col))
        column_index = non_zero_column_index[index_of_column_index][0]

        zero_colum_indexes = np.argwhere(column_projection == 0)
        left_side = zero_colum_indexes[zero_colum_indexes < column_index]
        left = np.max(left_side) if left_side.size > 0 else 0
        right_side = zero_colum_indexes[zero_colum_indexes > column_index]
        right = np.min(right_side) if right_side.size > 0 else self.resize_width
        return image[int(top*self.resize_height_ratio):int(bottom*self.resize_height_ratio), int(left*self.resize_width_ratio):int(right*self.resize_width_ratio)] 

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
