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

        if robot_name == "go2":
            xml_file = "assets/go2.xml"
            abdction_torque_limits: list = [-0.863, 0.863]
            hip_torque_limits: list = [-0.686, 4.501]
            knee_torque_limits: list = [-2.818, -0.888]

            self.action_space = gym.spaces.Box(
                low = np.array(
                    [abdction_torque_limits[0], hip_torque_limits[0], knee_torque_limits[0]]*4,
                    dtype=np.float32,
                ),
                high = np.array(
                    [abdction_torque_limits[1], hip_torque_limits[1], knee_torque_limits[1]]*4, 
                    dtype=np.float32,
                ),
            )
            self.action = np.zeros(shape=(12))
            self.previous_action = np.zeros(shape=(12))

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        self.key = self.model.key("home").id
        self.model.opt.timestep = 0.01

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(95,), dtype=np.float64
        )

        self.track_commands = np.array([0.5, 0.0, 0.0])
        self.max_episode_length = 1000
        self.episode = 0
        self.steps_until_next_cmd = 0

        self.render_mode = render_mode

        if self.render_mode == "human":
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, title=f"Legged Mujoco: {robot_name}, Task: {task_name}", width=width, height=height)
            self._init_viewer_config()

        self.reward_scales = {
            ## Tracking
            "tracking_lin_vel": 3.0,
            "tracking_ang_vel": 0.8,
            ## Base
            "lin_vel_z": -2.0,
            "angvel_xy": -0.05,
            "orientation": -5.0,
            ## Other
            "termination": -1.0,
            "stand_still": -0.5,
            "default_pose": 2.0,
            ## Regularization
            "torques": -0.00025,
            "action_rate": -0.01,
            ## Feet
            "feet_air_time": 0.2,
            ## Special
            "tracking_sigma": 0.25,
            "max_foot_height": 0.1,
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

        self.torso_body_id = self.model.body("trunk").id
                        
    def _get_obs(self):
        gyro_id = self.model.sensor("gyro").adr[0]
        gyro = self.data.sensordata[gyro_id:gyro_id+3]
        noisy_gyro = (
            gyro
            + (2 * np.random.uniform(size=gyro.shape)-1)
            * self.noise_scales["level"] * self.noise_scales["gyro"]
        )

        gravity = self.projected_gravity()
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

        linvel_id = self.model.sensor("local_linvel").adr[0]
        linvel = self.data.sensordata[linvel_id:linvel_id+3]
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

        accelerometer_id = self.model.sensor("accelerometer").adr[0]
        accelerometer = self.data.sensordata[accelerometer_id:accelerometer_id+3]

        global_angvel_id = self.model.sensor("global_angvel").adr[0]
        global_angvel = self.data.sensordata[global_angvel_id:global_angvel_id+3]

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
        upvector = self.model.sensor("upvector").adr[0]
        fall_termination = data.sensordata[upvector: upvector+3][-1] < 0.0
        return fall_termination
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        qpos = self.init_pos
        qvel = np.zeros(self.model.nv)

        dxy = np.random.uniform(low=-0.5, high=0.5, size=(2, ))
        qpos[0:2] += dxy

        self.data.qpos = qpos
        qvel[0:6] += np.random.uniform(low=-0.5, high=0.5, size=(6,))
        self.data.qvel = qvel

        # mujoco.mj_resetDataKeyframe(self.model, self.data, self.key)
        mujoco.mj_forward(self.model, self.data)

        self.episode = 0
        self.feet_air_time = np.zeros(4)
        self.last_contacts = np.zeros(4, dtype=bool)
        self.previous_action = np.zeros(shape=(12))

        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        target_action = action*self.action_scale + self.default_joint_position
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

        reward = self.reward(self.data, action, done, first_contact)
        reward = reward*self.model.opt.timestep

        terminated, truncated = False, False

        if done:
            terminated = True

        if self.episode >= self.max_episode_length:
            truncated = True

        if self.steps_until_next_cmd % 500_000 == 0:
            self.sample_command()

        if self.render_mode == "human":
            self.render()
        
        self.previous_action = action
        self.last_contacts = contact
        self.feet_air_time *= ~contact
        return observation, reward, terminated, truncated, info
    
    def reward(self, data, action, done, first_contact):
        linvel_track_penalty    = self._reward_tracking_lin_vel(data)
        angvel_track_penalty    = self._reward_tracking_ang_vel(data)
        base_linvelz_penalty    = self._reward_lin_vel_z(data)
        angvel_xy_penalty       = self._reward_angvel_xy(data)
        orientation_penalty     = self._reward_orientation(data)
        stand_still_penalty     = self._reward_stand_still(data.qpos[7:])
        termination_penalty     = self._reward_termination(done)
        default_pose_penalty    = self._reward_default_joint_pos(data.qpos[7:])
        joint_torque_penalty    = self._reward_motor_torque(data.actuator_force)
        action_rate_penalty     = self._reward_action_rate(action)
        feet_air_time_penalty   = self._reward_feet_air_time(first_contact)

        rewards = self.reward_scales['tracking_lin_vel']    *linvel_track_penalty   + \
                self.reward_scales['tracking_ang_vel']      *angvel_track_penalty   + \
                self.reward_scales['lin_vel_z']             *base_linvelz_penalty   + \
                self.reward_scales['angvel_xy']             *angvel_xy_penalty      + \
                self.reward_scales['orientation']           *orientation_penalty    + \
                self.reward_scales['stand_still']           *stand_still_penalty    + \
                self.reward_scales['termination']           *termination_penalty    + \
                self.reward_scales['default_pose']          *default_pose_penalty   + \
                self.reward_scales['torques']               *joint_torque_penalty   + \
                self.reward_scales['action_rate']           *action_rate_penalty    + \
                self.reward_scales['feet_air_time']         *feet_air_time_penalty 
        
        return rewards
    
    def sample_command(self):
        lin_vel_x = np.random.uniform(0.1, 1.0)
        lin_vel_y = np.random.uniform(0.1, 0.5)
        self.track_commands = np.array([lin_vel_x, lin_vel_y, 0.0])

    def projected_gravity(self):
        imu_id = self.model.site("imu").id
        rot_matrix = self.data.site_xmat[imu_id].reshape((3, 3))
        return rot_matrix.T @ np.array([0, 0, -1])

    def get_foot_contacts(self):
        touch_indices = [4, 7, 10, 13]
        feet_contact_forces = self.data.cfrc_ext[touch_indices]
        return np.linalg.norm(feet_contact_forces, axis=1)
    
    ## Tracking Rewards
    
    def _reward_tracking_lin_vel(self, data):
        local_linvel_adr = self.model.sensor("local_linvel").adr[0]
        local_linvel = data.sensordata[local_linvel_adr: local_linvel_adr+3]
        lin_vel_error = np.sum(np.square(self.track_commands[:2] - local_linvel[:2]))
        return np.exp(-lin_vel_error/self.reward_scales['tracking_sigma'])
    
    def _reward_tracking_ang_vel(self, data):
        gyro_adr = self.model.sensor("gyro").adr[0]
        gyro = data.sensordata[gyro_adr: gyro_adr+3]
        ang_vel_error = np.square(self.track_commands[2] - gyro[2])
        return np.exp(-ang_vel_error/self.reward_scales['tracking_sigma'])
    
    ## Base Rewards

    def _reward_lin_vel_z(self, data):
        global_linvel_adr = self.model.sensor("global_linvel").adr[0]
        global_linvel = data.sensordata[global_linvel_adr: global_linvel_adr+3]
        return np.square(global_linvel[2])
    
    def _reward_angvel_xy(self, data):
        global_angvel_adr = self.model.sensor("global_angvel").adr[0]
        global_angvel = data.sensordata[global_angvel_adr: global_angvel_adr+3]
        return np.sum(np.square(global_angvel[:2]))
    
    def _reward_orientation(self, data):
        torso_zaxis_adr = self.model.sensor("upvector").adr[0]
        torso_zaxis = data.sensordata[torso_zaxis_adr: torso_zaxis_adr+3]
        return np.sum(np.square(torso_zaxis[:2]))
    
    ## Regularization Rewards

    def _reward_motor_torque(self, torques):
        return np.sqrt(np.sum(np.square(torques))) + np.sum(np.abs(torques))

    def _reward_action_rate(self, act):
        return np.sum(np.square(act - self.previous_action))
    
    ## Other Rewards

    def _reward_default_joint_pos(self, qpos):
        weight = np.array([1.0, 1.0, 0.1] * 4)
        return np.exp(-np.sum(np.square(qpos - self.default_joint_position) * weight))
    
    def _reward_stand_still(self, qpos):
        cmd_norm = np.linalg.norm(self.track_commands)
        return np.sum(np.abs(qpos - self.default_joint_position)) * (cmd_norm < 0.01)

    def _reward_termination(self, done):
        return done
    
    ## Feet Rewards
    def _reward_feet_air_time(self, first_contact):
        desired_air_time = 0.1
        cmd_norm = np.linalg.norm(self.track_commands)
        rew_airtime = np.sum((self.feet_air_time - desired_air_time)*first_contact)
        rew_airtime *= cmd_norm > 0.01
        return rew_airtime
    
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
