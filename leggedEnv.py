import numpy as np

import mujoco
import mujoco_viewer
import gymnasium as gym
from typing import Optional

from datetime import datetime
from rich.console import Console

from scipy.spatial.transform import Rotation as R

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

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(89,), dtype=np.float64
        )

        self.action_low = np.array(self.model.jnt_range[1:].T[0], dtype=np.float64)
        self.action_high = np.array(self.model.jnt_range[1:].T[1], dtype=np.float64)

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
        self.steps_until_next_cmd = 0
        self.simulate_action_latency = False

        self.render_mode = render_mode

        if self.render_mode == "human":
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, title=f"Legged Mujoco: {robot_name}, Task: {task_name}", width=width, height=height)
            self._init_viewer_config()

        self.reward_scales = {
            ## Tracking
            "tracking_lin_vel": 2.5,
            "tracking_ang_vel": 0.8,
            ## Base
            "lin_vel_z": -1.0,
            "angvel_xy": -0.1,
            "base_height": -50.0,
            ## Other
            "default_pose": -0.1,
            ## Regularization
            "torques": -2e-5,
            "action_rate": -0.1,
            ## Special
            "tracking_sigma": 0.05,
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

        self.action_scale = 0.25
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
    
    def _get_info(self):
        base_pos = self.data.xpos[1]

        return {
            "distance": np.linalg.norm(
                base_pos, ord=2
            )
        }
    
    def _get_termination(self, data):
        sleep_termination = data.xpos[self.trunk_id][-1] < 0.2
        return sleep_termination 
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key)
        mujoco.mj_forward(self.model, self.data)

        self.episode = 0
        self.feet_air_time = np.zeros(4)
        self.last_contacts = np.zeros(4, dtype=bool)
        self.previous_action = np.zeros(shape=(12))

        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        action_apply = self.previous_action if self.simulate_action_latency else action
        target_action = action_apply*self.action_scale + self.default_joint_position
        self.data.ctrl[:] = target_action
        mujoco.mj_step(self.model, self.data)
        self.episode += 1
        self.steps_until_next_cmd += 1

        contact = self.get_foot_contacts() > 1.0
        contact_filt = contact | self.last_contacts
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.model.opt.timestep
        p_f = self.data.site_xpos[self.foot_site_id]
        p_fz = p_f[..., -1]

        r = R.from_quat([self.data.xquat[1][1], self.data.xquat[1][2], self.data.xquat[1][3], self.data.xquat[1][0]])
        roll, pitch, _ = r.as_euler('xyz', degrees=True)
        bad_state = True if abs(roll) > self.roll_threshold or abs(pitch) > self.pitch_threshold else False

        observation = self._get_obs()
        info = self._get_info()
        done = self._get_termination(self.data) or bad_state

        reward = self.reward(self.data, action)
        terminated, truncated = False, False

        if done:
            terminated = True

        if self.episode >= self.max_episode_length:
            truncated = True

        # if self.steps_until_next_cmd % 500_000 == 0:
        #     self.sample_command()

        if self.render_mode == "human":
            self.render()
        
        self.previous_action = action
        self.last_contacts = contact
        self.feet_air_time *= ~contact
        return observation, reward, terminated, truncated, info
    
    def reward(self, data, action):
        linvel_track_penalty    = self._reward_tracking_lin_vel(data)
        angvel_track_penalty    = self._reward_tracking_ang_vel(data)
        base_linvelz_penalty    = self._reward_lin_vel_z(data)
        angvel_xy_penalty       = self._reward_angvel_xy(data)
        base_height_penalty     = self._reward_base_height(data)
        default_pose_penalty    = self._reward_default_joint_pos(data.qpos[7:])
        joint_torque_penalty    = self._reward_motor_torque(data.actuator_force)
        action_rate_penalty     = self._reward_action_rate(action)

        rewards = self.reward_scales['tracking_lin_vel']    *linvel_track_penalty   + \
                self.reward_scales['tracking_ang_vel']      *angvel_track_penalty   + \
                self.reward_scales['lin_vel_z']             *base_linvelz_penalty   + \
                self.reward_scales['angvel_xy']             *angvel_xy_penalty      + \
                self.reward_scales['base_height']           *base_height_penalty    + \
                self.reward_scales['default_pose']          *default_pose_penalty   + \
                self.reward_scales['torques']               *joint_torque_penalty   + \
                self.reward_scales['action_rate']           *action_rate_penalty        
        return rewards
    
    def sample_command(self):
        lin_vel_x = np.random.uniform(0.1, 1.0)
        lin_vel_y = np.random.uniform(0.1, 0.5)
        self.track_commands = np.array([lin_vel_x, lin_vel_y, 0.0])

    def get_foot_contacts(self):
        touch_indices = [4, 7, 10, 13]
        feet_contact_forces = self.data.cfrc_ext[touch_indices]
        return np.linalg.norm(feet_contact_forces, axis=1)
    
    ## Tracking Rewards
    
    def _reward_tracking_lin_vel(self, data):
        local_linvel = data.sensordata[self.localvel_id: self.localvel_id+3]
        lin_vel_error = np.sum(np.square(self.track_commands[:2] - local_linvel[:2]))
        return np.exp(-lin_vel_error/self.reward_scales['tracking_sigma'])
    
    def _reward_tracking_ang_vel(self, data):
        gyro = data.sensordata[self.gyro_id: self.gyro_id+3]
        ang_vel_error = np.square(self.track_commands[2] - gyro[2])
        return np.exp(-ang_vel_error/self.reward_scales['tracking_sigma'])
    
    ## Base Rewards

    def _reward_lin_vel_z(self, data):
        linvel = data.sensordata[self.localvel_id: self.localvel_id+3]
        return np.square(linvel[2])
    
    def _reward_angvel_xy(self, data):
        angvel = data.sensordata[self.gyro_id: self.gyro_id+3]
        return np.sum(np.square(angvel[:2]))
    
    def _reward_base_height(self, data, base_height_target=0.304):
        base_height = data.xpos[self.trunk_id][-1]
        return np.square(base_height - base_height_target)
     
    ## Regularization Rewards

    def _reward_motor_torque(self, torques):
        return np.sum(np.abs(torques))

    def _reward_action_rate(self, act):
        return np.sum(np.square(act - self.previous_action))
    
    ## Other Rewards
    def _reward_default_joint_pos(self, qpos):
        return np.sum(np.abs(qpos - self.default_joint_position))
    
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
        print(f"Reward: {reward}")
        if done:
            obs, info = env.reset()