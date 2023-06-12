import gym
import numpy as np
import collections
import cv2
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace


class MSE(gym.Wrapper):
    """
        Role: To adjust the output game scene of Mario game, because when playing the game, the game continuity will make most of the continuous images acquired in a period of time are approximate.
             In order to reduce the approximate game scene image input and improve the training efficiency, we iterate through the skip frame images, calculate the sum of the rewards of this skip frame image, and do a maximum pooling in the last 2 frames of this skip frame.
             After pooling, an image is generated to represent the skip frame. Here skip=4
    """

    def __init__(self, env=None, skip=4):
        super(MSE, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self.buffer_obs = collections.deque(maxlen=2)
        self.skip = skip

    def step(self, action):

        reward_total = 0.0
        done = None
        # Iterate over skip frames, which is the same action over and over again to get an updated picture of the game
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            self.buffer_obs.append(obs)
            reward_total += reward
            if done:
                break
        # Compare the last 2 frames and select the one with the largest pixel value at the same position in both images as the new image,
        # which is the maximum pooling operation
        max_frame = np.max(np.stack(self.buffer_obs), axis=0)
        return max_frame, reward_total, done, info

    # After resetting the game, clear the container for the last 2 frames of the buffer.

    def reset(self):
        self.buffer_obs.clear()

        obs = self.env.reset()
        self.buffer_obs.append(obs)
        return obs


# the resize the shape of observation space, the original is too big
class MR84x84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(MR84x84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return MR84x84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
            # image normalization on RBG
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


# transfer from 84*84*1 to 1*84*84
class imgToTorch(gym.ObservationWrapper):

    def __init__(self, env):
        super(imgToTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


'''
Role: returns successive n_steps frames (after skip)

Implementation:
        When env calls the reset() method, it initializes self.buffer, initializing its size to n_steps*84*84.
        This means that four 1*84*84 game frames can be placed, the first three being all zeros when initialised, and the last one being the initialised game frame
'''


class BW(gym.ObservationWrapper):

    def __init__(self, env, n_steps, dtype=np.float32):
        super(BW, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)

        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]

        self.buffer[-1] = observation
        return self.buffer


class PixelNormalization(gym.ObservationWrapper):

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def get_super_mario_bros_env(envName='SuperMarioBros-1-1-v0', actions=None):
    if actions is None:
        actions = {"RIGHT_ONLY": RIGHT_ONLY}
    env = gym_super_mario_bros.make(envName)
    env = MSE(env)
    env = MR84x84(env)
    env = imgToTorch(env)
    env = BW(env, 4)
    env = PixelNormalization(env)
    env = JoypadSpace(env, actions)
    return env
