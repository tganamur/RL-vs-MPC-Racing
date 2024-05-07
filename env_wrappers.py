import gym
import numpy as np
from f110_gym.envs.base_classes import Integrator
from f110_gym.envs.f110_env import F110Env
from stable_baselines3.common.vec_env import DummyVecEnv
import yaml
from argparse import Namespace


work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 1.375}
# Load configuration
with open('f1tenth_racetracks\example\config_example_map.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)

class F110RLWrapper(gym.Wrapper):
    def __init__(self, env, planner, max_steps=5000):
        super().__init__(env)
        self.env = env
        self.planner = planner
        self.max_steps = max_steps
        self.step_count = 0
        self.conf = conf
        self.obs = None

        # Define the observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, -np.pi, 0.0]),
            high=np.array([np.inf, np.inf, np.pi, np.inf]),
            shape=(4,)
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            shape=(1,)
        )

    def reset(self):
        self.step_count = 0
        initial_pose = np.array([[conf.sx, conf.sy, conf.stheta]])
        self.obs, _, _, _ = self.env.reset(initial_pose)  # Assign obs here
        return self._preprocess_observation(self.obs)

    def step(self, action):
        action = self._preprocess_action(action)
        self.obs, reward, done, info = self.env.step(action)
        self.step_count += 1
        done = done or self.step_count >= self.max_steps
        return self._preprocess_observation(self.obs), reward, done, info

    def _preprocess_observation(self, obs):
        pose_x, pose_y, pose_theta = obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]
        linear_vel = obs['linear_vels_x'][0]
        return np.array([pose_x, pose_y, pose_theta, linear_vel], dtype=np.float32)

    def _preprocess_action(self, action):
        steer = action[0]
        pose_x, pose_y, pose_theta = self.obs['poses_x'][0], self.obs['poses_y'][0], self.obs['poses_theta'][0]
        speed, _ = self.planner.plan(pose_x, pose_y, pose_theta, work['tlad'], work['vgain'])
        return np.array([[steer, speed]], dtype=np.float32)

def make_env(env_id, planner, max_steps=5000):
    env = gym.make(env_id, map='f1tenth_racetracks\\example\\example_map', map_ext='.png', num_agents=1, timestep=0.01, integrator=Integrator.RK4)
    env = F110RLWrapper(env, planner, max_steps)
    return env 