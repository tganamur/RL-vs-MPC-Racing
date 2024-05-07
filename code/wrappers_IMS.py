# MIT License

# Copyright (c) 2020 FT Autonomous Team One

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import gym
import numpy as np
import pandas as pd

from gym import spaces
from pathlib import Path

from code.random_trackgen import create_track, convert_track
import csv
from scipy.spatial import cKDTree

#mapno = ["Austin","BrandsHatch","Budapest","Catalunya","Hockenheim","IMS","Melbourne","MexicoCity","Montreal","Monza","MoscowRaceway",
  #       "Nuerburgring","Oschersleben","Sakhir","SaoPaulo","Sepang","Shanghai","Silverstone","Sochi","Spa","Spielberg","YasMarina","Zandvoort"]
mapno = ['IMS']
#mapno = ['example']

randmap = mapno[0]
globwaypoints = np.genfromtxt(f"./f1tenth_racetracks/{randmap}/{randmap}_centerline.csv", delimiter=',')
df_c = pd.DataFrame(globwaypoints, columns=['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m'])
df2_c = df_c.drop(columns=['w_tr_right_m', 'w_tr_left_m'])

centerline = df2_c.to_numpy()
#raceline_waypoints = np.genfromtxt(f"./f1tenth_racetracks/{randmap}/{randmap}_raceline.csv", delimiter=';')

raceline_csv = np.genfromtxt(f"./f1tenth_racetracks/{randmap}/{randmap}_raceline.csv", delimiter=';')

df = pd.DataFrame(raceline_csv, columns=['s_m', 'x_m', 'y_m', 'psi_rad', 'kappa_rad', 'vx', 'ax'])
df2 = df.drop(columns=['s_m', 'kappa_rad', 'vx', 'ax'])

raceline = df2.to_numpy()

def convert_range(value, input_range, output_range):
    # converts value(s) from range to another range
    # ranges ---> [min, max]
    (in_min, in_max), (out_min, out_max) = input_range, output_range
    in_range = in_max - in_min
    out_range = out_max - out_min
    return (((value - in_min) * out_range) / in_range) + out_min

class F110_Wrapped(gym.Wrapper):
    """
    This is a wrapper for the F1Tenth Gym environment intended
    for only one car, but should be expanded to handle multi-agent scenarios
    """

    def __init__(self, env):
        super().__init__(env)

        # normalised action space, steer and speed
        self.action_space = spaces.Box(low=np.array(
            [-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float)

        # normalised observations, just take the lidar scans
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1080,), dtype=np.float)
        # Track the furthest point reached by the car
        self.furthest_point = None

        # store allowed steering/speed/lidar ranges for normalisation
        self.s_min = self.env.params['s_min']
        self.s_max = self.env.params['s_max']
        self.v_min = self.env.params['v_min']
        self.v_max = self.env.params['v_max']
        self.lidar_min = 0
        self.lidar_max = 30  # see ScanSimulator2D max_range

        # store car dimensions and some track info
        self.car_length = self.env.params['length']
        self.car_width = self.env.params['width']
        self.track_width = 3.2  # ~= track width, see random_trackgen.py

        # radius of circle where car can start on track, relative to a centerpoint
        self.start_radius = (self.track_width / 2) - \
            ((self.car_length + self.car_width) / 2)  # just extra wiggle room

        self.step_count = 0

        # set threshold for maximum angle of car, to prevent spinning
        self.max_theta = 100
        self.count = 0

        self.prev_vels = np.zeros(3)
        self.prev_steer_angle = 0.0
        self.prev_yaw = 0.0
        self._current_waypoint = np.zeros(2)
        self._current_index = 0
        self.prev_waypoint = np.zeros(2)

        self.waypoint_tree = cKDTree(raceline[:, :2]) 

    def _find_nearest_waypoint(self, agent_x, agent_y):
        """
        Find the nearest waypoint to the agent's current position using the KD-tree.
        """
        _, nearest_index = self.waypoint_tree.query([agent_x, agent_y])
        return raceline[nearest_index]

    def step(self, action):
        # convert normalised actions (from RL algorithms) back to actual actions for simulator
        action_convert = self.un_normalise_actions(action)
        observation, _, done, info = self.env.step(np.array([action_convert]))

        self.step_count += 1

        # TODO -> do some reward engineering here and mess around with this
        reward = 0

        # Get the agent's current position
        agent_x = observation['poses_x'][0]
        agent_y = observation['poses_y'][0]
        agent_theta = observation['poses_theta'][0]

        # Find the nearest waypoint to the agent's current position
        nearest_waypoint = self._find_nearest_waypoint(agent_x, agent_y)

        # Interpolate between nearest waypoint and the next waypoint
        _, nearest_index = np.where(raceline == nearest_waypoint)
        near_x = raceline[nearest_index][0][0]
        near_y = raceline[nearest_index][0][1]
        near_theta = raceline[nearest_index][0][2]
        dist = np.sqrt((agent_x-near_x)**2 + (agent_y-near_y)**2)
        heading_diff = np.abs(agent_theta - near_theta)
        #print('distance to raceline', dist)

        # Check if the current distance exceeds the furthest point reached
        distance_to_nearest_waypoint = np.linalg.norm([agent_x - nearest_waypoint[0], agent_y - nearest_waypoint[1]])
        if self.furthest_point is None or distance_to_nearest_waypoint > np.linalg.norm([agent_x - self.furthest_point[0], agent_y - self.furthest_point[1]]):
            self.furthest_point = nearest_waypoint

        dist_threshold = self.car_width/2.5
        heading_threshold = 0.1
        if dist > dist_threshold:
            reward += 50
        else: 
            reward -= 20

        if heading_diff < heading_threshold:
            reward += 25
        else: 
            reward -= 15

        angular_speed = observation['ang_vels_z'][0]
        # Get the agent's current steering angle
        current_steer_angle = action[0]

        # Penalize the difference between the current and previous steering angles
        steer_angle_diff = abs(current_steer_angle - self.prev_steer_angle)

        # Penalize sharp changes in steering more heavily to encourage smoother driving
        if steer_angle_diff > 0.1:  # Adjust threshold as needed
            reward -= 5 * steer_angle_diff ** 2  # Quadratic penalty for larger changes
        else:
            reward -= steer_angle_diff

        # Update previous steering angle for the next step
        self.prev_steer_angle = current_steer_angle

        
        # reward function that returns percent of lap left, maybe try incorporate speed into the reward too
        waypoints = np.genfromtxt(f"./f1tenth_racetracks/{randmap}/{randmap}_centerline.csv", delimiter=',')

        if self.count < len(centerline)-1:
            min_dist = self.car_width/1.5
            x = centerline[self.count][0]
            y = centerline[self.count][1]
            dist = np.sqrt((agent_x - x)**2 + (agent_y - y)**2)
            if dist < min_dist:
                reward += 5
            else: 
                reward -= 0.7
        else:
            self.count = 0
        
         # Calculate the distance to the next waypoint
        if self.count < len(centerline) - 1:
            next_x = centerline[self.count + 1][0]
            next_y = centerline[self.count + 1][1]
            next_dist = np.sqrt((agent_x - next_x)**2 + (agent_y - next_y)**2)

            # Add a small positive reward for getting closer to the next waypoint
            prev_dist = np.sqrt((observation['poses_x'][0] - centerline[self.count][0])**2 +
                                (observation['poses_y'][0] - centerline[self.count][1])**2)
            if next_dist < prev_dist:
                reward += 50

        # #eoins reward function
        # Calculate modified reward based on velocity magnitude
        vel_magnitude = np.linalg.norm([observation['linear_vels_x'][0], observation['linear_vels_y'][0]])
        # Adjust the reward based on velocity magnitude
        if vel_magnitude < 5:
        # Reward for low velocity
            reward += 0.8
        elif vel_magnitude > 20:
        # Penalize high velocity
            reward -= 5
        else:
        # Linearly interpolate reward between low and high velocity thresholds
            reward += (vel_magnitude - 10) * (0.08 / (20 - 5))

        if observation['collisions'][0]:
            self.count = 0
            reward -= 10

        # end episode if car is spinning
        if abs(observation['poses_theta'][0]) > self.max_theta:
            done = True
        reward -= 0.2

        if self.env.lap_counts[0] > 0:
            self.count = 0
            reward += 300  # or any large positive value
            self.env.lap_counts[0] = 0

        # Penalize high angular velocity (smoothness)
        ang_magnitude = abs(observation['ang_vels_z'][0])
        if ang_magnitude > 1:  # Adjust threshold as needed
            reward -= 2 * ang_magnitude ** 2  # Quadratic penalty for higher angular velocity


        return self.normalise_observations(observation['scans'][0]), reward, bool(done), info

    def reset(self, start_xy=None, direction=None):
        # should start off in slightly different position every time
        # position car anywhere along line from wall to wall facing
        # car will never face backwards, can face forwards at an angle

        # start from origin if no pose input
        if start_xy is None:
            start_xy = np.zeros(2)
        # start in random direction if no direction input
        if direction is None:
            direction = 3.14
        # get slope perpendicular to track direction
        slope = np.tan(direction + np.pi / 2)
        # get magintude of slope to normalise parametric line
        magnitude = np.sqrt(1 + np.power(slope, 2))
        # get random point along line of width track
        rand_offset = 0
        rand_offset_scaled = rand_offset * self.start_radius

        # convert position along line to position between walls at current point
        x, y = start_xy + rand_offset_scaled * np.array([1, slope]) / magnitude

        # point car in random forward direction, not aiming at walls
        t = direction
        # reset car with chosen pose
        observation, _, _, _ = self.env.reset(np.array([[x, y, t]]))
        # reward, done, info can't be included in the Gym format
        # Reset furthest point tracker
        self.furthest_point = None
        return self.normalise_observations(observation['scans'][0])

    def un_normalise_actions(self, actions):
        # convert actions from range [-1, 1] to normal steering/speed range
        steer = convert_range(actions[0], [-1, 1], [self.s_min, self.s_max])
        speed = convert_range(actions[1], [-1, 1], [self.v_min, self.v_max])
        return np.array([steer, speed], dtype=np.float)

    def normalise_observations(self, observations):
        # convert observations from normal lidar distances range to range [-1, 1]
        return convert_range(observations, [self.lidar_min, self.lidar_max], [-1, 1])

    def update_map(self, map_name, map_extension, update_render=True):
        self.env.map_name = map_name
        self.env.map_ext = map_extension
        self.env.update_map(f"{map_name}.yaml", map_extension)
        if update_render and self.env.renderer:
            self.env.renderer.close()
            self.env.renderer = None

    def seed(self, seed):
        self.current_seed = seed
        np.random.seed(self.current_seed)
        print(f"Seed -> {self.current_seed}")


class RandomMap(gym.Wrapper):
    """
    Generates random maps at chosen intervals, when resetting car,
    and positions car at random point around new track
    """

    # stop function from trying to generate map after multiple failures
    MAX_CREATE_ATTEMPTS = 20

    def __init__(self, env, step_interval=5000):
        super().__init__(env)
        # initialise step counters
        self.step_interval = step_interval
        self.step_count = 0

    def reset(self):
        # check map update interval
        if self.step_count % self.step_interval == 0:
            # create map
            for _ in range(self.MAX_CREATE_ATTEMPTS):
                try:
                    track, track_int, track_ext = create_track()
                    convert_track(track,
                                  track_int,
                                  track_ext,
                                  self.current_seed)
                    break
                except Exception:
                    print(
                        f"Random generator [{self.current_seed}] failed, trying again...")
            # update map
            self.update_map(f"./maps/map{self.current_seed}", ".png")
            # store waypoints
            self.waypoints = np.genfromtxt(f"centerline/map{self.current_seed}.csv",
                                           delimiter=',')
        # get random starting position from centerline
        random_index = np.random.randint(len(self.waypoints))
        start_xy = self.waypoints[random_index]
        print(start_xy)
        next_xy = self.waypoints[(random_index + 1) % len(self.waypoints)]
        # get forward direction by pointing at next point
        direction = np.arctan2(next_xy[1] - start_xy[1],
                               next_xy[0] - start_xy[0])
        # reset environment
        return self.env.reset(start_xy=start_xy, direction=direction)

    def step(self, action):
        # increment class step counter
        self.step_count += 1
        # step environment
        return self.env.step(action)

    def seed(self, seed):
        # seed class
        self.env.seed(seed)
        # delete old maps and centerlines
        for f in Path('centerline').glob('*'):
            if not ((seed - 100) < int(''.join(filter(str.isdigit, str(f)))) < (seed + 100)):
                try:
                    f.unlink()
                except:
                    pass
        for f in Path('maps').glob('*'):
            if not ((seed - 100) < int(''.join(filter(str.isdigit, str(f)))) < (seed + 100)):
                try:
                    f.unlink()
                except:
                    pass

class RandomF1TenthMap(gym.Wrapper):
    """
    Places the car in a random map from F1Tenth
    """

    # stop function from trying to generate map after multiple failures
    MAX_CREATE_ATTEMPTS = 20

    def __init__(self, env, step_interval=5000):
        super().__init__(env)
        # initialise step counters
        self.step_interval = step_interval
        self.step_count = 0

    def reset(self):
        # check map update interval
        if self.step_count % self.step_interval == 0:
            # update map
            randmap = mapno[0]
            #self.update_map(f"./maps/map{self.current_seed}", ".png")
            self.update_map(f"./f1tenth_racetracks/{randmap}/{randmap}_map", ".png")
            # store waypoints
            #self.waypoints = np.genfromtxt(f"centerline/map{self.current_seed}.csv",delimiter=',')
            self.waypoints = np.genfromtxt(f"./f1tenth_racetracks/{randmap}/{randmap}_centerline.csv", delimiter=',')
            globwaypoints = self.waypoints

        # get random starting position from centerline
        random_index = 0
        start_xy = self.waypoints[random_index]  #len = 4
        start_xy = start_xy[:2]
        next_xy = self.waypoints[(random_index + 1) % len(self.waypoints)]
        # get forward direction by pointing at next point
        direction = np.arctan2(next_xy[1] - start_xy[1],
                               next_xy[0] - start_xy[0])
        # reset environment
        return self.env.reset(start_xy=start_xy, direction=direction)

    def step(self, action):
        # increment class step counter
        self.step_count += 1
        # step environment
        return self.env.step(action)

    def seed(self, seed):
        # seed class
        self.env.seed(seed)
        # delete old maps and centerlines
        for f in Path('centerline').glob('*'):
            if not ((seed - 100) < int(''.join(filter(str.isdigit, str(f)))) < (seed + 100)):
                try:
                    f.unlink()
                except:
                    pass
        for f in Path('maps').glob('*'):
            if not ((seed - 100) < int(''.join(filter(str.isdigit, str(f)))) < (seed + 100)):
                try:
                    f.unlink()
                except:
                    pass


class ThrottleMaxSpeedReward(gym.RewardWrapper):
    """
    Slowly increase maximum reward for going fast, so that car learns
    to drive well before trying to improve speed
    """

    def __init__(self, env, start_step, end_step, start_max_reward, end_max_reward=None):
        super().__init__(env)
        # initialise step boundaries
        self.end_step = end_step
        self.start_step = start_step
        self.start_max_reward = start_max_reward
        # set finishing maximum reward to be maximum possible speed by default
        self.end_max_reward = self.v_max if end_max_reward is None else end_max_reward

        # calculate slope for reward changing over time (steps)
        self.reward_slope = (self.end_max_reward - self.start_max_reward) / (self.end_step - self.start_step)

    def reward(self, reward):
        # maximum reward is start_max_reward
        if self.step_count < self.start_step:
            return min(reward, self.start_max_reward)
        # maximum reward is end_max_reward
        elif self.step_count > self.end_step:
            return min(reward, self.end_max_reward)
        # otherwise, proportional reward between two step endpoints
        else:
            return min(reward, self.start_max_reward + (self.step_count - self.start_step) * self.reward_slope)
