import argparse
import glob
import logging
import time
import torch
from tqdm import tqdm
import pickle
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython import display
from super_mario_env import get_super_mario_bros_env
from agent import DDQNAgent


def get_logs_path(logs_root):
    os.makedirs(logs_root, exist_ok=True)
    files = glob.glob(os.path.join(logs_root, "exp*"), recursive=False)
    logs_dir = os.path.join(logs_root, f"exp{len(files)}")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    return logs_dir



logger = logging.getLogger(__name__)
# creating the logger to record our training
def logger_setting(logs_dir):

    log_file = os.path.join(logs_dir, "run.log")
    logger.setLevel(logging.DEBUG)


    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)


    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)


    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)


    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def log_and_update_progress(info):
    logger.info(info)

    # tqdm.write(info)


# the parameters used for training
def train(training_mode,
          num_episodes=1000,
          actions=RIGHT_ONLY,
          exploration_max=1,
          exploration_min=0.02,
          exploration_decay=0.001,
          lr=0.00025,
          batch_size=32,
          max_memory_size=30000,
          gamma=0.95,
          copy_step=5000,
          save_path="./",
          test_model_dir="",
          desc="",
          render_mode=0
          ):
    '''
        parameters：
            training_mode：True or False
                    True：the current mode is training mode
                    False：testing mode
            num_episodes: the number of episodes for training
            actions：RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
            exploration_max：0-1:  maximum probability of taking random actions, the range is from 0 to 1
            exploration_min: 0-1:  maximum probability of taking random actions, the range is from 0 to 1
            exploration_decay: exp_rate = exp_rate * exp_decay
            lr:learning rate for the model
            batch_size: the size of replay experience for learning
            max_memory_size: the size of replay memory
            gamma: the parameter used in DDQN equation
            copy_step: copy the weights from local net to target net after the number of copy steps
            save_path：the path to save the model
            test_model_dir: when training mode is False, this is the direction of loading the network for testing
            desc：the progress of training
            render_mode：the mode of showing mario image, 0 is showing local, 1 is showing on ipynp, 2 is not showing
    '''

    # loading the environment, if 'SuperMarioBros-1-1-v0', we only train on level 1-1
    env = get_super_mario_bros_env('SuperMarioBros-1-1-v0', actions)

    observation_space = env.observation_space.shape
    action_space = env.action_space.n

    # initialize our Agent for training
    agent = DDQNAgent(state_space=observation_space,
                      action_space=action_space,
                      max_memory_size=max_memory_size,
                      batch_size=batch_size,
                      gamma=gamma,
                      lr=lr,
                      exploration_max=exploration_max,
                      exploration_min=exploration_min,
                      exploration_decay=exploration_decay,
                      test=not training_mode,
                      test_models_dir=test_model_dir,
                      copy_step=copy_step
                      )


    env.reset()
    rewards_total = []
    average_rewards_total = {}

    for ep_num in tqdm(range(num_episodes), desc=desc):
        state = env.reset()  # reset the environment at the beginning of each episode
        state = torch.Tensor([state])  # transfer (4*84*84) numpy to a tensor for our network
        reward_total = 0
        steps = 0
        get_act_time = []
        get_train_time = []
        while True:
            # to decide if we are showing the mario image
            if render_mode == 0:
                env.render()
            elif render_mode == 1:
                plt.figure(3)
                plt.clf()
                plt.imshow(env.render(mode='rgb_array'))
                plt.title("Episode: %d" % (ep_num))
                plt.axis('off')
                display.clear_output(wait=True)
                display.display(plt.gcf())
            t1 = time.perf_counter()
            action = agent.act(state)
            t2 = time.perf_counter()
            steps += 1
            state_next, reward, terminal, info = env.step(int(action[0]))

            reward_total += reward
            state_next = torch.Tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)
            terminal = torch.tensor([int(terminal)]).unsqueeze(0)

            # if the current mode is training, we are learning from the replay experience
            if training_mode:
                t3 = time.perf_counter()
                agent.remember(state, action, reward, state_next, terminal)
                agent.experience_replay()
                t4 = time.perf_counter()
                get_train_time.append(t4 - t3)
            get_act_time.append(t2 - t1)

            state = state_next
            if terminal:
                break
        rewards_total.append(int(reward_total))

        if ep_num != 0 and ep_num % 1 == 0:
            # print("Episode {}, total score = {}, average score = {}".format(ep_num + 1, rewards_total[-1], np.mean(rewards_total)))

            log_and_update_progress(
                "Episode {}, total score={}, average score={:.2f}, exploration_rate={:.4f}, 每步耗时：{:.4f}, 训练耗时：{:.4f}".format(
                    ep_num + 1,
                    rewards_total[-1],
                    np.mean(rewards_total),
                    agent.exploration_rate,
                    sum(get_act_time) / len(get_act_time) if len(get_act_time) else 0,
                    sum(get_train_time) / len(get_train_time) if len(get_train_time) else 0))

            # plot the plot of total reward vs num of episodes
            if training_mode:
                average_rewards_total[ep_num] = np.mean(rewards_total)
                plt.title("episodes & average reward (DDQN)")
                plt.plot(list(average_rewards_total.keys()), list(average_rewards_total.values()))
                plt.savefig(os.path.join(save_path, "average_reward_plot.jpg"))
                plt.clf()

                with open(os.path.join(save_path, "rewards_total.pkl"), "wb") as f:
                    pickle.dump(rewards_total, f)
                with open(os.path.join(save_path, "average_rewards_total.pkl"), "wb") as f:
                    pickle.dump(average_rewards_total, f)

                torch.save(agent.local_net.state_dict(), os.path.join(save_path, "local_net.pt"))
                torch.save(agent.target_net.state_dict(), os.path.join(save_path, "target_net.pt"))

        num_episodes += 1

    log_and_update_progress(
        "Episode {}, total score = {}, average score = {}, exploration_rate = {}".format(ep_num + 1, rewards_total[-1],
                                                                                         np.mean(rewards_total),
                                                                                         agent.exploration_rate))
    # save the final model
    if training_mode:
        average_rewards_total[ep_num + 1] = np.mean(rewards_total)

        plt.title("episodes & average reward (DDQN)")
        plt.plot(list(average_rewards_total.keys()), list(average_rewards_total.values()))
        plt.savefig(os.path.join(save_path, "average_reward_plot.jpg"))
        plt.clf()

        with open(os.path.join(save_path, "rewards_total.pkl"), "wb") as f:
            pickle.dump(rewards_total, f)
        with open(os.path.join(save_path, "average_rewards_total.pkl"), "wb") as f:
            pickle.dump(rewards_total, f)

        torch.save(agent.local_net.state_dict(), os.path.join(save_path, "local_net.pt"))
        torch.save(agent.target_net.state_dict(), os.path.join(save_path, "target_net.pt"))

    env.close()

    return average_rewards_total


# get parameters from the command line, also we can initialize the parameters here
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False, help='True: training mode False: testing mode ')
    parser.add_argument('--test_model_dir', type=str, default='runs/exp16', help='if testing mode, load the model from directory')
    parser.add_argument('--num_episodes', type=int, default=1000, help='number of episodes')
    parser.add_argument('--batch_size', type=int, default=32, help='to get the batch size from experience for learning')
    parser.add_argument('--exploration_max', type=float, default=0.02, help='initial exp rate')
    parser.add_argument('--exploration_min', type=float, default=0.02, help='the minimum exp rate')
    parser.add_argument('--exploration_decay', type=float, default=0.999, help='decay')
    parser.add_argument('--lr', type=float, default=0.00025, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.90, help='how important the future is')
    parser.add_argument('--max_memory_size', type=int, default=100000, help='the size of replay experience')
    parser.add_argument('--copy_step', type=int, default=5000, help='copy the weight from local to target network after copy step')
    parser.add_argument('--actions', type=int, default=0, choices=[0, 1, 2],
                        help='three different action spaces 0: RIGHT_ONLY 1:SIMPLE_MOVEMENT 2:COMPLEX_MOVEMENT')
    parser.add_argument('--save_path', type=str, default='./runs', help='the path to save the training model')
    parser.add_argument('--desc', type=str, default='', help='the training progress')
    parser.add_argument('--render_mode', type=int, default=0,
                        help='to show the Mario image  0:show local 1:show on ipynp 3:not show')
    return parser.parse_args()


# start training
def start():
    opt = parse_opt()
    actions_list = [RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT]
    training_mode = opt.train
    num_episodes = opt.num_episodes
    actions = actions_list[opt.actions]
    exploration_max = opt.exploration_max
    exploration_min = opt.exploration_min
    exploration_decay = opt.exploration_decay
    lr = opt.lr
    batch_size = opt.batch_size
    max_memory_size = opt.max_memory_size
    gamma = opt.gamma
    copy_step = opt.copy_step
    save_path = opt.save_path
    test_model_dir = opt.test_model_dir
    desc = opt.desc
    render_mode = opt.render_mode

    logs_dir = get_logs_path(save_path)

    logger_setting(logs_dir)
    logger.info(f"\n"
                f"  training parameters：\n"
                f"     training mode：{training_mode}\n"
                f"     num_episodes：{num_episodes}\n"
                f"     actions：{opt.actions}\n"
                f"     exploration_max: {exploration_max}\n"
                f"     exploration_min: {exploration_min}\n"
                f"     exploration_decay: {exploration_decay}\n"
                f"     lr: {lr}\n"
                f"     batch size: {batch_size}\n"
                f"     max_memory_size: {max_memory_size}\n"
                f"     gamma: {gamma}\n"
                f"     copy_step: {copy_step}\n"
                f"     save_path: {logs_dir}\n"
                f"     test_model_dir: {test_model_dir}\n"
                f"     desc: {desc}\n"
                f"     render_mode: {render_mode}\n")
    # a=(training_mode,num_episodes,actions,exploration_max,exploration_min,exploration_decay,lr,batch_size,max_memory_size,gamma,copy_step,save_path,test_model_dir,desc,render_mode)
    # start training
    train(training_mode=training_mode,
          num_episodes=num_episodes,
          actions=actions,
          exploration_max=exploration_max,
          exploration_min=exploration_min,
          exploration_decay=exploration_decay,
          lr=lr,
          batch_size=batch_size,
          max_memory_size=max_memory_size,
          gamma=gamma,
          copy_step=copy_step,
          save_path=logs_dir,
          test_model_dir=test_model_dir,
          desc=desc,
          render_mode=render_mode
          )


if __name__ == '__main__':
    start()

