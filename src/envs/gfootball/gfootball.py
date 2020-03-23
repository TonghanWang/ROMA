from ..multiagentenv import MultiAgentEnv
from operator import attrgetter
from copy import deepcopy
from absl import flags
import numpy as np
import pygame
import sys
import os
import math
import time
import numpy as np

import gfootball.env as football_env
from gfootball.env import config, wrappers

'''
Google Football
'''


class GF(MultiAgentEnv):
    def __init__(self, **kwargs):
        num_agents = 2  # num_agents from kwargs

        self.env = football_env.create_environment(
            # env_name='test_example_multiagent',
            env_name='2_vs_2pre',  # env_name='3_vs_GK',
            representation='extracted',
            rewards='scoring',
            stacked=False,
            logdir='/tmp/rllib_test',
            write_goal_dumps=True,
            write_full_episode_dumps=True,
            render=True,
            dump_frequency=0,
            number_of_left_players_agent_controls=num_agents,
            channel_dimensions=(42, 42))  # the preferred size for many professional teams' stadiums is 105 by 68 metres

            # channel_dimensions=(3, 3))
        #self.env = wrappers.Simple115StateWrapper(self.env)

        self.n_agents = num_agents
        self.episode_limit = 300

        self.obs = None

        self.current_step_num = -1
        # the agent will get reward based on the distance between the ball and targeted goal.
        self.distance_reward = True
        self.discount_on_episode_limit = True
        # how the distance_reward should be discounted,
        # the value of distance_reward is in range [0. , self.distance_reward_discount_factor]
        self.distance_reward_discount_factor = 1.0
        # in order to not punish early shooting, we need to accumulate the reward for all left time steps and
        # reward such player with this reward. Basically,
        # self.accumulate_reward_on_score = builder.config().end_episode_on_score in your scenario.
        self.accumulate_reward_on_score = True
        #TODO:
        self.general_multiplier = 12
        self.owned_by_other_team_reward = -0.4
        self.ball_owned_team = -1  # -1 means no one, 0 means player team
        self.ball_owned_player = -1  # -1 means no one
        # if the ball was owned by the player team, and the ball was passed from one player to the other player,
        # add some punishment. Basically we want the player to be greedy.
        self.pocession_change_reward = -0.1 * (self.episode_limit ** int(self.discount_on_episode_limit))

    def step(self, actions):
        """ Returns reward, terminated, info """
        self.current_step_num += 1
        # len(reward) == num of agents.
        # info stores {'score_reward': int}
        observation, reward, done, info = self.env.step(actions)
        self.obs = observation

        # the customized reward (based on CheckpointRewardWrapper):
        if self.distance_reward:
            observation2 = self.env.unwrapped.observation()
            for rew_index in range(len(reward)):
                o = observation2[rew_index]
                # reward[rew_index] == 1 means that there is a goal from player rew_index. If there is a goal, we
                # add the reward for every player.
                if self.accumulate_reward_on_score and reward[rew_index] == 1:
                    reward[rew_index] += (self.episode_limit - self.current_step_num) / (self.episode_limit ** int(self.discount_on_episode_limit))
                    continue

                if o['ball_owned_team'] != -1 and o['ball_owned_team'] != 0:
                    reward[rew_index] += self.owned_by_other_team_reward / (self.episode_limit ** int(self.discount_on_episode_limit))
                    continue

                if self.ball_owned_player != -1 and self.ball_owned_player != o['ball_owned_player']:
                    self.ball_owned_player = o['ball_owned_player']
                    reward[rew_index] += self.pocession_change_reward / (self.episode_limit ** int(self.discount_on_episode_limit))
                    continue

                # Check if the active player has the ball.
                if ('ball_owned_team' not in o or
                        o['ball_owned_team'] != 0 or
                        'ball_owned_player' not in o or
                        o['ball_owned_player'] != o['active']):
                    continue

                # o['ball'][0] is X, in the range [-1, 1]. o['ball'][1] is Y, in the range [-0.42, 0.42]
                # (2*2+0.42*0.42)**0.5 = 2.0436242316042352
                # the closer d to zero means the closer it is to the (enemy or right???) team's gate
                d = ((o['ball'][0] - 1) ** 2 + o['ball'][1] ** 2) ** 0.5
                # we divide by self.episode_limit since this reward is accumulative, we don't want the accumulative
                # reward to be too large after all.
                reward[rew_index] += (1 - d/2.0436242316042352) * self.distance_reward_discount_factor / (self.episode_limit ** int(self.discount_on_episode_limit))
                # print("player",rew_index,":",d)
                # print(reward)
                # print(o['ball'])
        # print("o['ball_owned_team']")
        # print(o['ball_owned_team'])
        # print("o['ball_owned_player']")
        # print(o['ball_owned_player'])
        # print("local reward")
        # print(np.sum(reward))
        return np.sum(reward * self.general_multiplier), done, info


    def get_obs(self):
        """ Returns all agent observations in a list """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        obs_agent = self.obs[agent_id].flatten()
        return obs_agent

    def get_obs_size(self):
        """ Returns the shape of the observation """
        # if obs_space is (2, 10, 10, 4) it returns (10, 10, 4)
        obs_size = np.array(self.env.observation_space.shape[1:])
        return int(obs_size.prod())

    def get_state(self):

        #print("Shape of state: %s" % str(state.shape))
        # TODO: difference between observation and state unclear from the google football github
        return self.obs.flatten()
        # return self.env.get_state()

    def get_state_size(self):
        """ Returns the shape of the state"""
        state_size = np.array(self.env.observation_space.shape)
        return int(state_size.prod())

    def get_avail_actions(self):
        """Gives a representation of which actions are available to each agent.
        Returns nested list of shape: n_agents * n_actions_per_agent.
        Each element in boolean. If 1 it means that action is available to agent."""
        # assumed that all actions are available.

        total_actions = self.get_total_actions()

        avail_actions = [[1]*total_actions for i in range(0, self.n_agents)]
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id.
        Returns a list of shape: n_actions of agent.
        Each element in boolean. If 1 it means that action is available to agent."""
        # assumed that all actions are available.
        return [1]*self.get_total_actions()

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take.
        Should be integer of number of actions of an agent. Assumed that all agents have same number of actions."""
        return self.env.action_space.nvec[0]

    def get_stats(self):
        #raise NotImplementedError
        return {}

    def get_agg_stats(self, stats):
        return {}

    def reset(self):
        """ Returns initial observations and states"""
        self.obs = self.env.reset()  #.reshape(self.n_agents)
        self.current_step_num = -1
        # print("Shape of raw observations: %s" % str(self.obs.shape))
        # should be return self.get_obs(), self.get_state()
        return self.get_obs(), self.get_state()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self):
        pass
        # raise NotImplementedError

    def save_replay(self):
        pass
        # raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
