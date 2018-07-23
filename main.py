import argparse
import math
from collections import namedtuple
from itertools import count
from tqdm import tqdm
from tensorboardX import SummaryWriter
import sys
import gym
import numpy as np
from gym import wrappers
import os
import torch
from ddpg import DDPG
from naf import NAF
from normalized_actions import NormalizedActions
from ounoise import OUNoise
from param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from replay_memory import ReplayMemory, Transition
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from envs import make_env
import glob
import time
from visualize import visdom_plot
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--algo', default='DDPG',
                    help='algorithm to use: DDPG | NAF')
parser.add_argument('--env-name', required=True,
                    help='name of the environment to run')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='discount factor for model (default: 0.001)')
parser.add_argument('--ou_noise', type=bool, default=True)
parser.add_argument('--param_noise', type=bool, default=False)
parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                    help='initial noise scale (default: 0.3)')
parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                    help='final noise scale (default: 0.3)')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=4, metavar='N',
                    help='random seed (default: 4)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=1000, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--discrete', action='store_true', default=False, help='Use discrete version of DDPG')
parser.add_argument('--actor-lr', type=float, default=1e-4, help='actor lr')
parser.add_argument('--critic-lr', type=float, default=1e-3, help='critic lr')
parser.add_argument('--num-frames', type=int, default=1e7, help='number of frames')
parser.add_argument('--num-processes', type=int, default=15, help='number of parallel agents generating rollouts')
parser.add_argument('--log-dir', type=str, default='results', help='log directory name')
parser.add_argument('--update-fq', type=int, default=1, help='how often to update parameters')
parser.add_argument('--log-interval', type=int, default=50, help='log interval')
parser.add_argument('--vis', action='store_true', default=False, help='visdom visualization')
parser.add_argument('--no-ln', action='store_true', default=False, help='No layer normalization')
parser.add_argument('--warmup', type=int, default=10000, help='Number of insertions before updates')

args = parser.parse_args()

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

if not args.discrete:
    env = NormalizedActions(gym.make(args.env_name))
else:
    env = [make_env(args.env_name, args.seed, i, args.log_dir, False) for i in range(args.num_processes)]
    env = SubprocVecEnv(env)
#writer = SummaryWriter()

if args.vis:
    from visdom import Visdom
    viz = Visdom(port=8097, server='http://eos11')
    win = None

#env.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
if args.algo == "NAF":
    agent = NAF(args.gamma, args.tau, args.hidden_size,
                      env.observation_space.shape[0], env.action_space)
else:
    agent = DDPG(args.gamma, args.tau, args.hidden_size,
                      env.observation_space.shape[0], env.action_space, args.discrete, (args.actor_lr, args.critic_lr), args)

memory = ReplayMemory(args.replay_size)

if not args.discrete:
    ounoise = OUNoise(env.action_space.shape[0]) if args.ou_noise else None
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05, 
        desired_action_stddev=args.noise_scale, adaptation_coefficient=1.05) if args.param_noise else None
else:
    ounoise = None
    param_noise = None

rewards = []
total_numsteps = 0
updates = 0
device = torch.device('cuda')
num_steps = args.num_frames // args.num_processes

state = torch.Tensor(env.reset())

episode_rewards = torch.zeros(args.num_processes, 1).to(device)
final_rewards = torch.zeros(args.num_processes, 1).to(device)
start = time.time()
for step in range(int(num_steps)):
    '''
    if args.ou_noise and ounoise is not None: 
        ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                      i_episode) / args.exploration_end + args.final_noise_scale
        ounoise.reset()

    if args.param_noise and args.algo == "DDPG" and param_noise is not None:
        agent.perturb_actor_parameters(param_noise)
    '''
    episode_reward = 0
    state = state.to(device)
    action, action_probs, entropy = agent.select_action(state, ounoise, param_noise)
    #print(action.cpu().numpy())
    if args.discrete:
       use_action = action.squeeze(1).cpu().numpy()
    else:
        use_action = action.cpu().numpy()[0]
    next_state, reward, done, _ = env.step(use_action)
    total_numsteps += 1
    episode_reward += reward

    #action = torch.LongTensor(action)
    reward = torch.Tensor(reward).to(device).unsqueeze(1)
    episode_rewards += reward
    mask = torch.Tensor([[1.0] if not x else [0.0] for x in done]).to(device)
    final_rewards *= mask
    final_rewards += (1 - mask) * episode_rewards
    episode_rewards *= mask

    next_state = torch.Tensor(next_state).to(device)
    for i in range(args.num_processes):
        if args.discrete:
            memory.push(state[i], action_probs[i], mask[i], next_state[i], reward[i])
        else:
            memory.push(state[i], action[i], mask[i], next_state[i], reward[i])

    state = next_state

    if len(memory) > args.warmup and step % args.update_fq == 0:
        for _ in range(args.updates_per_step):
            transitions = memory.sample(args.batch_size)
            batch = Transition(*zip(*transitions))

            value_loss, policy_loss = agent.update_parameters(batch)

            #writer.add_scalar('loss/value', value_loss, updates)
            #writer.add_scalar('loss/policy', policy_loss, updates)

            updates += 1

    #writer.add_scalar('reward/train', episode_reward, i_episode)

    # Update param_noise based on distance metric
    if args.param_noise and param_noise is not None:
        episode_transitions = memory.memory[memory.position-t:memory.position]
        states = torch.cat([transition[0] for transition in episode_transitions], 0)
        unperturbed_actions = agent.select_action(states, None, None)
        perturbed_actions = torch.cat([transition[1] for transition in episode_transitions], 0)

        ddpg_dist = ddpg_distance_metric(perturbed_actions.numpy(), unperturbed_actions.numpy())
        param_noise.adapt(ddpg_dist)

    rewards.append(episode_reward)
    if step % args.log_interval == args.log_interval - 1 and len(memory) > args.warmup:
        '''
        state = torch.Tensor([env.reset()])
        episode_reward = 0
        while True:
            action, action_probs = agent.select_action(state.to(device), explore=False)
            if args.discrete:
                use_action = action.item()
            else:
                use_action = action.cpu().numpy()[0]
            next_state, reward, done, _ = env.step(use_action)
            episode_reward += reward

            next_state = torch.Tensor([next_state])

            state = next_state
            if done:
                break

        #writer.add_scalar('reward/test', episode_reward, i_episode)
        '''
        end = time.time()
        total_num_steps = step * args.num_processes
        rewards.append(episode_reward)
        #print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode, total_numsteps, rewards[-1], np.mean(rewards[-10:])))
        print("Num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, value loss {:.5f}, policy loss {:.5f}, entropy {:.5f}".
                format(total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), 
                       value_loss, policy_loss, entropy.item()))

    if args.vis and step % args.log_interval == 0 and len(memory) > args.warmup:
        try:
            win = visdom_plot(viz, win, args.log_dir, args.env_name, 'disc_ddpg', args.num_frames)
        except IOError:
            pass
env.close()
