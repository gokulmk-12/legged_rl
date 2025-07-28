import numpy as np
from utils import *

def euler_from_quaternion(w, x, y, z):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z

def reward_tracking_lin_vel(data, localvel_id, track_commands, reward_scales):
    local_linvel = data.sensordata[localvel_id: localvel_id+3]
    lin_vel_error = np.sum(np.square(track_commands[:2] - local_linvel[:2]))
    return np.exp(-lin_vel_error / reward_scales['tracking_sigma'])

def reward_tracking_ang_vel(data, gyro_id, track_commands, reward_scales):
    gyro = data.sensordata[gyro_id: gyro_id+3]
    ang_vel_error = np.square(track_commands[2] - gyro[2])
    return np.exp(-ang_vel_error/reward_scales['tracking_sigma'])

def reward_lin_vel_z(data, localvel_id):
    linvel = data.sensordata[localvel_id: localvel_id+3]
    return -np.square(linvel[2])

def reward_angvel_xy(data, gyro_id):
    angvel = data.sensordata[gyro_id: gyro_id+3]
    return -np.sum(np.square(angvel[:2]))

def reward_base_height(data, trunk_id, base_height_target=0.304):
    base_height = data.xpos[trunk_id][-1]
    return -np.abs(base_height - base_height_target)
    
def reward_motor_torque(torques):
    return - np.sqrt(np.sum(np.square(torques))) - np.sum(np.abs(torques))

# def reward_energy(data, torques):
#     return -np.sum(abs(data.qvel[6:]) * abs(torques))

def reward_action_smoothness(current_action, previous_action, previous_previous_action):
    c1 = np.sum(np.square(current_action - previous_action))
    c2 = np.sum(np.square(current_action - 2 * previous_action + previous_previous_action))
    return - c1 - c2

def reward_joint_acceleration(data):
    return -np.sum(np.square(data.qacc[6:]))

def reward_feet_airtime(air_time, first_contact, track_commands):
    cmd_norm = np.linalg.norm(track_commands)
    rew_air_time = np.sum((air_time-0.25) * first_contact)
    rew_air_time *= cmd_norm > 0.01
    return rew_air_time

def reward_default_pose(data, default_pose):
    return -np.sum(np.square(data.qpos[7:] - default_pose))

def reward_orientation(torso_zaxis):
    return -np.sum(np.square(torso_zaxis[:2]))

def reward_joint_velocity(data):
    joint_vel = data.qvel[6:]
    return - np.sum(np.square(joint_vel))

def reward_collision(data, calf_ids):
    contact_list = np.array([geoms_colliding(data, geom1=0, geom2=calf_id) for calf_id in calf_ids])
    return -10.0 * np.sum(contact_list)
