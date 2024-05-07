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

from code.random_trackgen import create_trafrom pyglet.gl import GL_POINTS
from numba import njit
from argparse import Namespace
import yaml

#mapno = ["Austin","BrandsHatch","Budapest","Catalunya","Hockenheim","IMS","Melbourne","MexicoCity","Montreal","Monza","MoscowRaceway",
  #       "Nuerburgring","Oschersleben","Sakhir","SaoPaulo","Sepang","Shanghai","Silverstone","Sochi","Spa","Spielberg","YasMarina","Zandvoort"]
#mapno = ['IMS']
mapno = ['example']

randmap = mapno[0]
# globwaypoints = np.genfromtxt(f"./f1tenth_racetracks/{randmap}/{randmap}_centerline.csv", delimiter=',')
# df_c = pd.DataFrame(globwaypoints, columns=['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m'])
# df2_c = df_c.drop(columns=['w_tr_right_m', 'w_tr_left_m'])

#centerline = df2_c.to_numpy()
raceline_waypoints = np.genfromtxt(f"./f1tenth_racetracks/{randmap}/{randmap}_waypoints.csv", delimiter=';')
with open('/Users/yash/ME292B_FinalProject/f1tenth_racetracks/example/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)

df = pd.DataFrame(raceline_waypoints, columns=['s_m', 'x_m', 'y_m', 'psi_rad', 'kappa_rad', 'vx', 'ax'])
df2 = df.drop(columns=['s_m', 'kappa_rad', 'vx', 'ax'])

raceline = df2.to_numpy()

def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns actuation
    """
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle


class PurePursuitPlanner:
    """
    Example Planner
    """
    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.

        self.drawn_waypoints = []

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """

        #points = self.waypoints

        points = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        
        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]
        
    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        gets the current waypoint to follow
        """
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """
        gives actuation given observation
        """
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed

        return speed, steering_angle

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
        self.conf = conf

        # Initialize the PurePursuitPlanner
        self.planner = PurePursuitPlanner(conf, self.car_length)
        self.lookahead_distance = 5.0

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

        # Use the PurePursuitPlanner to get the nearest waypoint and distance
        lookahead_point = self.planner._get_current_waypoint(raceline_waypoints, self.lookahead_distance, np.array([agent_x, agent_y]), agent_theta)
        if lookahead_point is not None:
            waypoint_x, waypoint_y, waypoint_speed = lookahead_point
            dist_to_waypoint = np.sqrt((agent_x - waypoint_x)**2 + (agent_y - waypoint_y)**2)
            reward += dist_to_waypoint  # Penalize distance from the waypoint

        #vel_magnitude = np.linalg.norm(
         #   [observation['linear_vels_x'][0], observation['linear_vels_y'][0]])
         #/10 maybe include if speed is having too much of an effect
        #if vel_magnitude > 5 and vel_magnitude < 20:
         #   reward -= 2
        
        #vel_magnitude = np.linalg.norm(
        #    [observation['linear_vels_x'][0], observation['linear_vels_y'][0]])
        #reward = vel_magnitude
            
            # Calculate modified reward based on velocity magnitude
        vel_magnitude = np.linalg.norm([observation['linear_vels_x'][0], observation['linear_vels_y'][0]])
        # Adjust the reward based on velocity magnitude
        if vel_magnitude < 5:
        # Reward for low velocity
            reward += 0.1
        elif vel_magnitude > 20:
        # Penalize high velocity
            reward -= 5
        else:
        # Linearly interpolate reward between low and high velocity thresholds
            reward += (vel_magnitude - 10) * (0.08 / (20 - 5))









            # Get the agent's current steering angle
        #current_steer_angle = action[0]

        # Penalize the difference between the current and previous steering angles
        #steer_angle_diff = abs(current_steer_angle - self.prev_steer_angle)
        #reward -= steer_angle_diff
        #self.prev_steer_angle = current_steer_angle
        
        # Get the agent's current steering angle
        current_steer_angle = action[0]

        # Penalize the difference between the current and previous steering angles
        steer_angle_diff = abs(current_steer_angle - self.prev_steer_angle)

        # Penalize sharp changes in steering more heavily to encourage smoother driving
        if steer_angle_diff > 0.1:  # Adjust threshold as needed
            reward -= 2 * steer_angle_diff
        else:
            reward -= steer_angle_diff

        # Update previous steering angle for the next step
        self.prev_steer_angle = current_steer_angle







        # reward function that returns percent of lap left, maybe try incorporate speed into the reward too
        #waypoints = np.genfromtxt(f"./f1tenth_racetracks/{randmap}/{randmap}_centerline.csv", delimiter=',')

        # if self.count < len(centerline)-1:
        #     min_dist = self.car_width/1.5
        #     x = centerline[self.count][0]
        #     y = centerline[self.count][1]
        #     dist = np.sqrt((agent_x - x)**2 + (agent_y - y)**2)
        #     if dist < min_dist:
        #         reward += 5
        #     else: 
        #         reward -= 0.7
        # else:
        #     self.count = 0
        
        # # Calculate the distance to the next waypoint
        # if self.count < len(centerline) - 1:
        #     next_x = centerline[self.count + 1][0]
        #     next_y = centerline[self.count + 1][1]
        #     next_dist = np.sqrt((agent_x - next_x)**2 + (agent_y - next_y)**2)

        #     # Add a small positive reward for getting closer to the next waypoint
        #     prev_dist = np.sqrt((observation['poses_x'][0] - centerline[self.count][0])**2 +
        #                         (observation['poses_y'][0] - centerline[self.count][1])**2)
        #     if next_dist < prev_dist:
        #         reward += 0.1

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
        
        if ang_magnitude > 3:
            reward -= 0.8

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
            direction = 0
        # get slope perpendicular to track direction
        slope = np.tan(direction + np.pi / 2)
        # get magintude of slope to normalise parametric line
        magnitude = np.sqrt(1 + np.power(slope, 2))
        # get random point along line of width track
        rand_offset = 0
        rand_offset_scaled = rand_offset * self.start_radius

        # convert position along line to position between walls at current point
        x, y = start_xy 

        # point car in random forward direction, not aiming at walls
        t = direction
        # reset car with chosen pose
        observation, _, _, _ = self.env.reset(np.array([[self.conf.sx, self.conf.sy, self.conf.stheta]]))
        
        # reward, done, info can't be included in the Gym format
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
        random_index = 0
        start_xy = self.waypoints[random_index]
        print(start_xy)
        next_xy = self.waypoints[(random_index + 1)]
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
            self.waypoints = np.genfromtxt(f"./f1tenth_racetracks/{randmap}/{randmap}_waypoints.csv", delimiter=';')
date_map(f"./maps/map{self.current_seed}", ".png")
            self.update_map(f"./f1tenth_racetracks/{randmap}/{randmap}_map", ".png")
            # store waypoints
            #self.waypoints = np.genfromtxt(f"centerline/map{self.current_seed}.csv",delimiter=',')
            self.waypoints = np.genfromtxt(f"./f1tenth_racetracks/{randmap}/{randmap}_centerline.csv", delimiter=',')
rrent_seed}", ".png")
            self.update_map(f"./f1tenth_racetracks/{randmap}/{randmap}_map", ".png")
            # store waypoints
            #self.waypoints = np.genfromtxt(f"centerline/map{self.current_seed}.csv",delimiter=',')
            self.waypoints = np.genfromtxt(f"./f1tenth_racetracks/{randmap}/{randmap}_centerline.csv", delimiter=',')
current_seed}", ".png")
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
