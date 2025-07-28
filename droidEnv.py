import mujoco
import numpy as np
import mujoco_viewer
import gymnasium as gym

from datetime import datetime
from rich.console import Console

def timestamp():
    return datetime.now().strftime("%H:%M:%S")

class DroidEnv(gym.Env):
    def __init__(self,
            robot_name: str = "droid",
            task_name: str = "walking",
            render_mode: str = "human",
            width: int = 700,
            height: int = 700     
        ):
        super(DroidEnv, self).__init__()

        self.robot_name = robot_name
        xml_file = "assets/droid.xml"

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        self.key = self.model.key("home").id
        self.model.opt.timestep = 0.01

        self.render_mode = render_mode
        if self.render_mode == "human":
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, title=f"Legged Mujoco: {robot_name}, Task: {task_name}", width=width, height=height)
            self._init_viewer_config()
            self.render()

        self._init_qpos = self.model.keyframe("home").qpos
        self.valid_action = list(range(0, 5)) + list(range(11, 16)) 
        self._default_pos = self.model.keyframe("home").qpos[7:][self.valid_action]

        self._lowers, self._uppers = self.model.jnt_range[1:6].T
        
        self.action_space = gym.spaces.Box(
            low = np.concatenate((self._lowers, self._lowers), dtype=np.float32),
            high = np.concatenate((self._uppers, self._uppers), dtype=np.float32),
        )
        self.action = np.zeros(shape=(self.action_space.shape))
        self.prev_action = np.zeros(shape=(self.action_space.shape))

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(48, ), dtype=np.float64
        )

        self.action_scale = 0.5
        self._command = np.array([])
        self.episode_length = 1000
        self.epsiode_count = 0

        self.console = Console()
        self.console.print(f"[Mujoco-Env] [{timestamp()}] Welcome to Legged Mujoco", style="green")
        self.console.print(f"[Mujoco-Env] [{timestamp()}] Robot: {robot_name}, Task: {task_name}", style="green")

    def _init_viewer_config(self,):
        self.viewer.cam.distance = 1.86
        self.viewer.cam.azimuth = 123.5
        self.viewer.cam.elevation = -13.75
        self.viewer.cam.lookat = np.array([0.07919824,-0.13157017,0.11765827])
    
    def _get_obs(self):
        gyro_id = self.model.sensor("gyro").adr[0]
        gyro = self.data.sensordata[gyro_id: gyro_id+3]

        imu_id = self.model.site("imu").id
        rot_matrix = self.data.site_xmat[imu_id].reshape((3, 3))
        gravity = rot_matrix.T @ np.array([0, 0, -1])

        joint_angle = self.data.qpos[7:][self.valid_action]

        joint_vel = self.data.qvel[6:][self.valid_action]

        linvel_id = self.model.sensor("local_linvel").adr[0]
        linvel = self.data.sensordata[linvel_id: linvel_id+3]

        accelerometer_id = self.model.sensor("accelerometer").adr[0]
        accelerometer = self.data.sensordata[accelerometer_id: accelerometer_id+3]

        global_angvel_id = self.model.sensor("global_angvel").adr[0]
        global_angvel = self.data.sensordata[global_angvel_id: global_angvel_id+3]

        com_height = self.data.qpos[2]

        state = np.hstack([
            gyro,
            accelerometer,
            gravity,
            linvel,
            global_angvel,
            joint_angle - self._default_pos,
            joint_vel,
            com_height,
            self.data.actuator_force,
        ])

        return state
    
    def _get_termination(self, data):
        upvector_id = self.model.sensor("upvector").adr[0]
        upvector = data.sensordata[upvector_id: upvector_id+3]
        fall_termination = upvector[-1] < 0.0
        return (
            fall_termination | np.isnan(self.data.qpos).any() | np.isnan(self.data.qvel).any()
        )

    def reset(self):
        qpos = self._init_qpos
        qvel = np.zeros(self.model.nv)

        self.data.qpos = qpos
        self.data.qvel = qvel

        mujoco.mj_forward(self.model, self.data)

        self.epsiode_count = 0
        self.prev_action = np.zeros(shape=(self.action_space.shape))

        observation = self._get_obs()
        return observation

    def step(self, action):
        target_action = self.action_scale * action + self._default_pos
        self.data.ctrl[:] = np.concatenate(target_action[0:5], np.zeros(shape=(6,)), target_action[5:])
        mujoco.mj_step(self.model, self.data)
        self.epsiode_count += 1

        obs = self._get_obs()
        done = self._get_termination(self.data)

        terminated, truncated = False, False

        if self.epsiode_count >= self.episode_length:
            truncated = True
        
        if self.render_mode == "human":
            self.render()
        
        self.prev_action = action
        reward = 0.0
        return obs, reward, terminated, truncated

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
    env = DroidEnv(robot_name="droid", task_name="walking", render_mode="human")

    obs = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated = env.step(action)
        if terminated:
            obs = env.reset()        