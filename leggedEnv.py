import numpy as np

import mujoco
import mujoco_viewer
import gymnasium as gym
from typing import Optional

from datetime import datetime
from rich.console import Console

from rewards import *

def timestamp():
    return datetime.now().strftime("%H:%M:%S")

console = Console()

class LeggedEnv(gym.Env):
    def __init__(self, 
            robot_name: str = "go2", 
            task_name: str = "walking",
            render_mode: str = "human",
            width: int = 1200, 
            height: int = 700
        ):
        super(LeggedEnv, self).__init__()

        self.robot_name = robot_name
        xml_file = f"assets/{robot_name}.xml"

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        self.key = self.model.key("home").id
        self.model.opt.timestep = 0.01
        self.ctrl_dt = 0.02

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(89,), dtype=np.float64
        )

        # self.action_low = np.array(self.model.jnt_range[1:].T[0], dtype=np.float64)
        # self.action_high = np.array(self.model.jnt_range[1:].T[1], dtype=np.float64)

        self.action_low = np.array([-0.3, 0.3, -1.9]*4)
        self.action_high = np.array([0.3, 1.15, -1.2]*4)

        self.action_space = gym.spaces.Box(
            low = self.action_low,
            high = self.action_high,
            shape=(12, ), dtype=np.float64
        )

        self.action = np.zeros(shape=(12))
        self.previous_action = np.zeros(shape=(12))

        self.track_commands = np.array([0.5, 0.0, 0.0])
        self.max_episode_length = 1000
        self.episode = 0
        self.info = {}
        self.steps_until_next_cmd = 0
        self.simulate_action_latency = False

        self.render_mode = render_mode

        if self.render_mode == "human":
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, title=f"Legged Mujoco: {robot_name}, Task: {task_name}", width=width, height=height)
            self._init_viewer_config()

        self.reward_scales = {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 1.0,
            "lin_vel_z": 0.2,
            "angvel_xy": 0.2,
            "base_height": 6.0,
            "torques": 5e-4,
            "action_rate": 0.2,
            "tracking_sigma": 0.25,
            "acceleration": 5e-6,
            "feet_air_time": 0.5,
            "default_pose": 0.2,
            "orientation": 1.0,
            "energy": 1e-3,
        }

        self.noise_scales = {
            "gyro": 0.1,
            "joint_pos": 0.03,
            "joint_vel": 0.1,
            "gravity": 0.05,
            "linvel": 0.1,
            "level": 1.0
        }

        console.print(f"[Mujoco-Env] [{timestamp()}] Welcome to Legged Mujoco", style="green")
        console.print(f"[Mujoco-Env] [{timestamp()}] Robot: {robot_name}, Task: {task_name}", style="green")

        self.action_scale = 0.0
        self.roll_threshold = 15
        self.pitch_threshold = 15

        self.feet_air_time = np.zeros(4)
        self.last_contacts = np.zeros(4, dtype=bool)
        self.init_pos = np.array(self.model.keyframe("home").qpos)
        self.default_joint_position = np.array(self.model.keyframe("home").qpos[7:])

        self._get_indices()

    def _init_viewer_config(self,):
        self.viewer.cam.distance = 1.86
        self.viewer.cam.azimuth = 123.5
        self.viewer.cam.elevation = -13.75
        self.viewer.cam.lookat = np.array([0.07919824,-0.13157017,0.11765827])
    
    def _get_indices(self):
        self.foot_site_id = []
        self.leg_names = ["FR", "FL", "RR", "RL"]
        foot_site_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        for name in foot_site_names:
            self.foot_site_id.append(self.model.site(name).id)

        self.trunk_id = self.model.body("trunk").id

        self.gyro_id = self.model.sensor("gyro").adr[0]
        self.localvel_id = self.model.sensor("local_linvel").adr[0]
        self.acc_id = self.model.sensor("accelerometer").adr[0]
        self.upvector_id = self.model.sensor("upvector").adr[0]
        self.global_linvel_id = self.model.sensor("global_linvel").adr[0]
        self.global_angvel_id = self.model.sensor("global_angvel").adr[0]
                        
    def _get_obs(self):
        gyro = self.data.sensordata[self.gyro_id:self.gyro_id+3]
        noisy_gyro = (
            gyro
            + (2 * np.random.uniform(size=gyro.shape)-1)
            * self.noise_scales["level"] * self.noise_scales["gyro"]
        )

        gravity = self.data.sensordata[self.upvector_id: self.upvector_id]
        noisy_gravity = (
            gravity
            + (2 * np.random.uniform(size=gravity.shape)-1)
            * self.noise_scales["level"] * self.noise_scales["gravity"]
        )

        joint_angles = self.data.qpos[7:]
        noisy_joint_angles = (
            joint_angles
            + (2 * np.random.uniform(size=joint_angles.shape) - 1)
            * self.noise_scales["level"] * self.noise_scales["joint_pos"]
        )

        joint_vel = self.data.qvel[6:]
        noisy_joint_vel = (
            joint_vel
            + (2 * np.random.uniform(size=joint_vel.shape) - 1)
            * self.noise_scales["level"] * self.noise_scales["joint_vel"]
        )

        linvel = self.data.sensordata[self.localvel_id:self.localvel_id+3]
        noisy_linvel = (
            linvel
            + (2 * np.random.uniform(size=linvel.shape) - 1)
            * self.noise_scales["level"] * self.noise_scales["linvel"]
        )

        state = np.hstack([
            noisy_linvel,
            noisy_gyro,
            noisy_gravity,
            noisy_joint_angles - self.default_joint_position,
            noisy_joint_vel,
            self.previous_action,
            self.track_commands,
        ])

        accelerometer = self.data.sensordata[self.acc_id:self.acc_id+3]
        global_angvel = self.data.sensordata[self.global_angvel_id:self.global_angvel_id+3]

        obs = np.concatenate(
            [
                state,
                gyro,
                accelerometer,
                gravity, 
                linvel,
                global_angvel,
                joint_angles - self.default_joint_position,
                joint_vel, 
                self.last_contacts,
                self.feet_air_time,
            ]
        )
        return obs
    
    def _get_termination(self, data):
        sleep_termination = data.xpos[self.trunk_id][-1] < 0.2
        return sleep_termination 
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key)
        mujoco.mj_forward(self.model, self.data)

        self.episode = 0
        self.feet_air_time = np.zeros(4)
        self.last_contacts = np.ones(4, dtype=bool)
        self.previous_action = np.zeros(shape=(12))

        observation = self._get_obs()
        return observation, {}
    
    def step(self, action):
        action_apply = self.previous_action if self.simulate_action_latency else action
        target_action = action_apply*self.action_scale + self.default_joint_position
        target_action = action_apply
        self.data.ctrl[:] = target_action
        mujoco.mj_step(self.model, self.data)
        self.episode += 1
        self.steps_until_next_cmd += 1

        contact = self.get_foot_contacts()
        contact_filt = contact | self.last_contacts
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.ctrl_dt
        p_f = self.data.site_xpos[self.foot_site_id]
        p_fz = p_f[..., -1]

        w, x, y, z = self.data.qpos[3:7]
        roll, pitch, _ = euler_from_quaternion(w, x, y, z)
        bad_state = True if abs(roll) > np.deg2rad(self.roll_threshold) or abs(pitch) > np.deg2rad(self.pitch_threshold) else False

        observation = self._get_obs()
        done = self._get_termination(self.data) or bad_state

        reward, info = self.reward(self.data, action, first_contact)
        terminated, truncated = False, False

        if done:
            terminated = True

        if self.episode >= self.max_episode_length:
            truncated, terminated = True, True

        # if self.steps_until_next_cmd % 500_000 == 0:
        #     self.sample_command()

        if self.render_mode == "human":
            self.render()
        
        self.previous_action = action
        self.last_contacts = contact
        self.feet_air_time *= ~contact
        return observation, reward, terminated, truncated, info
    
    def reward(self, data, action, first_contact):
        linvel_track        = reward_tracking_lin_vel(data, self.localvel_id, self.track_commands, self.reward_scales)
        angvel_track        = reward_tracking_ang_vel(data, self.gyro_id, self.track_commands, self.reward_scales)
        base_linvelz        = reward_lin_vel_z(data, self.localvel_id)
        angvel_xy           = reward_angvel_xy(data, self.gyro_id)
        base_height         = reward_base_height(data, self.trunk_id)
        joint_torque        = reward_motor_torque(data.qfrc_actuator[-12:])
        action_smoothness   = reward_action_smoothness(action, self.previous_action)
        joint_acceleration  = reward_joint_acceleration(data)
        feet_air_time       = reward_feet_airtime(self.feet_air_time, first_contact, self.track_commands)
        default_pose        = reward_default_pose(data, self.default_joint_position)
        orientation         = reward_orientation(data)
        energy              = reward_energy(data, data.qfrc_actuator[-12:])

        positive_rewards = self.reward_scales['tracking_lin_vel']    *linvel_track       + \
                self.reward_scales['tracking_ang_vel']               *angvel_track       + \
                self.reward_scales['feet_air_time']                  *feet_air_time
        
        negative_rewards = self.reward_scales['lin_vel_z']  *base_linvelz       + \
                self.reward_scales['angvel_xy']             *angvel_xy          + \
                self.reward_scales['base_height']           *base_height        + \
                self.reward_scales['torques']               *joint_torque       + \
                self.reward_scales['acceleration']          *joint_acceleration + \
                self.reward_scales['action_rate']           *action_smoothness  + \
                self.reward_scales['default_pose']          *default_pose       + \
                self.reward_scales['orientation']           *orientation        + \
                self.reward_scales['energy']                *energy
        
        rewards = positive_rewards * np.exp(0.02 * negative_rewards)

        info = {
            "reward_tracking_linvel":   linvel_track,
            "reward_tracking_angvel":   angvel_track,
            "reward_linvel_z":          base_linvelz,
            "reward_angvel_xy":         angvel_xy,
            "reward_base_height":       base_height,
            "reward_torques":           joint_torque,
            "reward_acceleration":      joint_acceleration,
            "reward_acton_smoothness":  action_smoothness,
            "reward_feet_airtime":      feet_air_time,
            "reward_default_pose":      default_pose,
            "reward_orientation":       orientation,
            "reward_energy":            energy,
        }
        return rewards, info
    
    def sample_command(self):
        lin_vel_x = np.random.uniform(-1.0, 1.0)
        lin_vel_y = np.random.uniform(0.1, 0.5)
        self.track_commands = np.array([lin_vel_x, 0.0, 0.0])

    def get_foot_contacts(self):
        touch_sensors = [self.model.sensor(f"{leg_name}_touch").adr[0] for leg_name in self.leg_names]
        contact_values = [bool(self.data.sensordata[touch_sensors[i]] > 0.0) for i in range(4)]
        return np.array(contact_values)
      
    def render(self):
        if self.viewer.is_alive:
            self.viewer.render()
        else:
            raise RuntimeError("Viewer is not initialized")
        
    def close(self):
        if self.render_mode == "human":
            if self.viewer.is_alive:
                self.viewer.close() 
        else:
            pass

if __name__ == "__main__":
    env = LeggedEnv(robot_name="go2", task_name="walking", render_mode="human")
    obs, info = env.reset()
  
    while True:
        action = env.action_space.sample()
        state, reward, done, _, _ = env.step(action)
        if done:
            obs, info = env.reset()