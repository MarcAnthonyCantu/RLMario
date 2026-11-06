import gym
import torch
import random, datetime, numpy as np
from skimage import transform

from gym.spaces import Box

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        resize_obs = transform.resize(observation, self.shape)
        # cast float back to uint8
        resize_obs *= 255
        resize_obs = resize_obs.astype(np.uint8)
        return resize_obs


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.time_penalty = 0.005
        self.step_count = 0
    
    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)

    def reward(self, reward):
        #Add bonus for reaching the flag
        if self.env.unwrapped._flag_get:
            in_game_time = self.env.unwrapped._time
            reward += (1/4)*in_game_time
            print("Flag reached!")
            
        
        #increasingly subtract reward as time passes
        self.step_count += 1
        reward -= self.time_penalty * min(self.step_count, 1000)
        return reward

class RewardClippingWrapper(gym.RewardWrapper):
    """
    Clips the raw reward from gym_super_mario_bros to prevent uint8 overflow
    artifacts from corrupting training. Place BEFORE SkipFrame in the wrapper
    chain so overflowed values don't get accumulated across skipped frames.
    """
    def __init__(self, env, lower=-15.0, upper=15.0):
        super().__init__(env)
        self.lower = lower
        self.upper = upper
    
    def reward(self, reward):
        # Detect uint8 wraparound: legitimate per-step rewards in this env
        # are in single digits. Anything above ~200 is almost certainly an
        # overflow (Mario moved backward, died, or transitioned to a new area).
        if reward > 200:
            reward = self.lower
        return float(np.clip(reward, self.lower, self.upper))
    
class ScoreRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_score = 0

    def reset(self, **kwargs):
        self.prev_score = 0
        return self.env.reset(**kwargs)

    def reward(self, reward):
        # Add reward for score gained
        score = getattr(self.env.unwrapped, '_score', 0)
        reward += score - self.prev_score
        self.prev_score = score
        #print('+', score - self.prev_score)
        return reward
    