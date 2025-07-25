import numpy as np

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
    return -np.square(base_height - base_height_target)
    
def reward_motor_torque(torques):
    return -np.sum(np.abs(torques))

def reward_action_smoothness(current_action, previous_action):
    return -np.sum(np.square(current_action - previous_action))

# def reward_default_joint_pos(qpos, default_joint_position):
#     return np.sum(np.abs(qpos - default_joint_position))