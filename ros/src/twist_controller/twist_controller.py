from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy


GAS_DENSITY = 2.858
ONE_MPH = 0.44704

MIN_THROTTLE = 0.0
MAX_THROTTLE = 0.5
MIN_SPEED = 0.1


# PID Controller params
kp = 0.15
ki = 0.0001
kd = 3.5


# LowPass filter params
tau = 0.5
ts = 0.02


class Controller(object):
    
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit,
            wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
              
        # Yaw controller for steering angles
        self.yaw_controller = YawController(wheel_base, steer_ratio, MIN_SPEED, max_lat_accel, max_steer_angle)
        
        # PID controller for throttle
        self.throttle_controller = PID(kp, ki, kd, MIN_THROTTLE, MAX_THROTTLE)     
        
        # Lowpass filter for smoothing noise 
        self.lowpass_filter = LowPassFilter(tau, ts)

        
        # Initialise internal variables 
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius   
        
        self.last_velocity = 0.0
        self.last_time = rospy.get_time()
        
        
    def control(self, dbw_enabled, current_velocity, angular_velocity, linear_velocity):
        
        # Reset PID controller and return zero values if car is in manual mode
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0
        
        # Apply lowpass filter to velocity data 
        current_velocity = self.lowpass_filter.filt(current_velocity)
        
        # Get steering from yaw controller
        steering = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_velocity)
        
        # Calculate velocity error and sample time, pass through to PID controller
        velocity_error = linear_velocity - current_velocity
        self.last_velocity = current_velocity
        
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        throttle = self.throttle_controller.step(velocity_error, sample_time)
        
        
        # Initialise brake as 0, update below based on vehicle condition
        brake = 0.0
        
        # If car isn't moving enough brake to keep car at a standstill
        if linear_velocity == 0 and current_velocity < MIN_SPEED:
            throttle = 0.0
            brake = self.vehicle_mass * self.wheel_radius
        
        # If velocity error is negative car is moving faster than needed; apply brake proportional to error size
        elif throttle < 0.1 and velocity_error < 0: 
            throttle = 0.0
            decel = max(velocity_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius
        
        # Return throttle, brake, steering
        return throttle, brake, steering
