#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math
import numpy as np
from scipy.spatial import KDTree 

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
'''

# Number of waypoints to publish
LOOKAHEAD_WPS = 50
PUBLISHING_RATE = 25
# Number of waypoints before stopline where car should come to rest
STOPLINE_BUFFER = 4
MAX_DECEL = 1.0


class WaypointUpdater(object):
    
    def __init__(self):
        
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)
        
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Member variables
        self.pose = None
        self.base_waypoints = None
        self.waypoints2d = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1
        
        self.loop()

        
    # Publish closest waypoint 
    def loop(self):
        
        rate = rospy.Rate(PUBLISHING_RATE)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoint_tree:
                self.publish_waypoints()
                rate.sleep()
    
    
    def get_closest_waypoint_idx(self):
        
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        
        # Get closest index from KDTree
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        
        # Check if closest point is in front or behind
        closest_vect = np.array(self.waypoints2d[closest_idx])
        prev_vect = np.array(self.waypoints2d[closest_idx - 1])
        
        pos_vect = np.array([x, y])
        
        # Check position of closest waypoint, positive sign means waypoint is behind 
        closest_direction = np.dot(closest_vect - prev_vect, pos_vect - closest_vect)

        # Adjust so waypoint is always ahead of current vehicle position 
        if closest_direction > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints2d)
        
        return closest_idx
    

    # Publish new list of waypoints from waypoint closest in front up to the lookahead amount
    def publish_waypoints(self):
        lane = self.generate_lane()
        self.final_waypoints_pub.publish(lane)
    
    
    def generate_lane(self):
        lane = Lane()
        closest_idx = self.get_closest_waypoint_idx()
        furthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_idx:furthest_idx]
        
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= furthest_idx):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
            
        return lane
   
     
    def decelerate_waypoints(self, base_waypoints, closest_idx):
        
        decel_waypoints = []
                
        for i, wp in enumerate(base_waypoints):
            
            p = Waypoint()
            p.pose = wp.pose
            
            # Add a buffer so the vehicle will avoid crossing the stop line
            stop_idx = max(self.stopline_wp_idx - closest_idx - STOPLINE_BUFFER, 0)
        
            dist = self.distance(base_waypoints, i, stop_idx)
            velocity = math.sqrt(MAX_DECEL * dist) + (i / LOOKAHEAD_WPS)
            if velocity < 1.0:
                velocity = 0.0 
            
            p.twist.twist.linear.x = min(velocity, wp.twist.twist.linear.x)
            decel_waypoints.append(p)
   
        return decel_waypoints      
    
        
    # Pose callback to save the current position
    def pose_cb(self, msg):
        self.pose = msg

        
    # Store base waypoints and extract positions to store in waypoint tree
    def waypoints_cb(self, waypoints):
        
        self.base_waypoints = waypoints
       
        if not self.waypoints2d:
            self.waypoints2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in                                              waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints2d)
    
    
    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg.data

   
    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    
    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

        
    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2  + (a.z - b.z)**2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node')
