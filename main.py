import argparse
parser = argparse.ArgumentParser(description='Set to debug mode')
parser.add_argument('--debug', type=bool, default=False, help='Set to debug mode')
args = parser.parse_args()


import debugpy
if args.debug:
    debugpy.listen(("0.0.0.0", 5678))  # Start the debug server
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()  # Pause execution until the debugger attaches


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                       module="gym_super_mario_bros")

import random, datetime
from pathlib import Path

from tqdm import tqdm
import numpy as np
import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace
from metrics import MetricLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame, RewardWrapper, RewardClippingWrapper

# Initialize Super Mario environment
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')


jump_only = JoypadSpace(
    env,
    [['right'],
    ['right', 'A']
    ]
)
jump_and_run = JoypadSpace(
    env,
    [['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B']
    ]
)
reward_eval = JoypadSpace(env, [[]])

env = jump_and_run

# Apply Wrappers to environment
env = RewardClippingWrapper(env)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
#env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)
env = RewardWrapper(env)
env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = Path('mario_net.chkpt') # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

episodes = 10000

### for Loop that train the model num_episodes times by playing the game
for e in tqdm(range(episodes)):

    state = env.reset()

    # Play the game!
    while True:

        # 3. Show environment
        env.render()

        # 4. Run agent on the state
        action = mario.act(state)

        # 5. Agent performs action
        next_state, reward, done, info = env.step(action)

        # 6. Remember
        mario.cache(state, next_state, action, reward, done)

        # 7. Learn
        q, loss = mario.learn()

        # 8. Logging
        logger.log_step(reward, loss, q)

        # 9. Update state
        state = next_state

        # 10. Check if end of game
        if info['flag_get']:
            break
        if done:
            break

        #print(reward)
    logger.log_episode()

    if e % 100 == 0:
        logger.record(
            episode=e,
            epsilon=mario.epsilon,
            step=mario.curr_step
        )
mario.save()
