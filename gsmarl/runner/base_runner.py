import copy
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter
from gsmarl.utils.graph_separated_buffer import GraphSeparatedReplayBuffer
from gsmarl.utils.util import update_linear_schedule


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.parameter_share = self.all_args.parameter_share

        # interval
        self.save_interval = self.all_args.save_interval
        self.log_interval = self.all_args.log_interval
        self.gamma = self.all_args.gamma
        self.use_popart = self.all_args.use_popart

        # model dir
        self.model_dir = self.all_args.model_dir
        self.save_gifs = self.all_args.save_gifs

        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        from gsmarl.algorithms.gsmarl.gsmarl import GSMARL as TrainAlgo
        from gsmarl.algorithms.gsmarl.algorithm.GSMARLPolicy import GSMARLPolicy as Policy
        self.policy = []
        if self.parameter_share:
            agent_id = 0
            share_observation_space = self.envs.share_observation_space[agent_id]
            po = Policy(self.all_args,
                self.envs.observation_space[agent_id],
                share_observation_space,
                self.envs.action_space[agent_id],
                device=self.device)
            for agent_id in range(self.num_agents):
                self.policy.append(po)
        else:
            for agent_id in range(self.num_agents):
                share_observation_space = self.envs.share_observation_space[agent_id]
                # policy network 
                po = Policy(self.all_args,
                            self.envs.observation_space[agent_id],
                            share_observation_space,
                            self.envs.action_space[agent_id],
                            device=self.device)
                self.policy.append(po)

        if self.model_dir is not None and self.all_args.restore_model:
            self.restore()

        self.trainer = []
        self.buffer = []

        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device=self.device)
            # buffer
            share_observation_space = self.envs.share_observation_space[agent_id]
            if self.n_rollout_threads == 1:
                bu = GraphSeparatedReplayBuffer(self.all_args,
                                        self.envs.envs[0].graph_obs_shape,
                                        self.envs.envs[0].share_graph_obs_shape,
                                        self.envs.action_space[agent_id])
            else:
                bu = GraphSeparatedReplayBuffer(self.all_args,
                                        self.envs.graph_obs_shape,
                                        self.envs.share_graph_obs_shape,
                                        self.envs.action_space[agent_id])
            # TODO: Shared buffer
            self.buffer.append(bu)
            self.trainer.append(tr)

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].agent_id[-1],
                                                                self.buffer[agent_id].share_nodes_feats[-1],
                                                                self.buffer[agent_id].share_edge_index[-1],
                                                                self.buffer[agent_id].share_edge_attr[-1],
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

            next_costs = self.trainer[agent_id].policy.get_cost_values(self.buffer[agent_id].agent_id[-1],
                                                                    self.buffer[agent_id].share_nodes_feats[-1],
                                                                    self.buffer[agent_id].share_edge_index[-1],
                                                                    self.buffer[agent_id].share_edge_attr[-1],
                                                                    self.buffer[agent_id].rnn_states_cost[-1],
                                                                    self.buffer[agent_id].masks[-1])
            next_costs = _t2n(next_costs)
            self.buffer[agent_id].compute_cost_returns(next_costs, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        cost_train_infos = []
        # random update order
        action_dim = self.buffer[0].actions.shape[-1]
        factor = np.ones((self.episode_length, self.n_rollout_threads, action_dim), dtype=np.float32)
        # randperm is better for NPS
        for agent_id in torch.randperm(self.num_agents):
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[
                                                                                   agent_id].available_actions.shape[
                                                                               2:])
            old_actions_logprob, _, _, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                self.buffer[agent_id].agent_id[:-1].reshape(-1, *self.buffer[agent_id].agent_id.shape[2:]),
                self.buffer[agent_id].nodes_feats[:-1].reshape(-1, *self.buffer[agent_id].nodes_feats.shape[2:]),
                self.buffer[agent_id].edge_index[:-1].reshape(-1, *self.buffer[agent_id].edge_index.shape[2:]),
                self.buffer[agent_id].edge_attr[:-1].reshape(-1, *self.buffer[agent_id].edge_attr.shape[2:]),
                self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))

            train_info = self.trainer[agent_id].train(self.buffer[agent_id])

            new_actions_logprob, dist_entropy, action_mu, action_std = self.trainer[agent_id].policy.actor.evaluate_actions(
                self.buffer[agent_id].agent_id[:-1].reshape(-1, *self.buffer[agent_id].agent_id.shape[2:]),
                self.buffer[agent_id].nodes_feats[:-1].reshape(-1, *self.buffer[agent_id].nodes_feats.shape[2:]),
                self.buffer[agent_id].edge_index[:-1].reshape(-1, *self.buffer[agent_id].edge_index.shape[2:]),
                self.buffer[agent_id].edge_attr[:-1].reshape(-1, *self.buffer[agent_id].edge_attr.shape[2:]),
                self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            

            factor = factor * _t2n(torch.exp(new_actions_logprob - old_actions_logprob).reshape(self.episode_length,
                                                                                                self.n_rollout_threads,
                                                                                                action_dim))
            train_infos.append(train_info)

            self.buffer[agent_id].after_update()

        return train_infos, cost_train_infos

    def save(self):
        agent_id = 0
        if self.parameter_share:
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer[agent_id].policy.cost_critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/cost_critic_agent" + str(agent_id) + ".pt")
        else:
            for agent_id in range(self.num_agents):
                policy_actor = self.trainer[agent_id].policy.actor
                torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
                policy_critic = self.trainer[agent_id].policy.critic
                torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")
                policy_critic = self.trainer[agent_id].policy.cost_critic
                torch.save(policy_critic.state_dict(), str(self.save_dir) + "/cost_critic_agent" + str(agent_id) + ".pt")

    def restore(self):
        agent_id = 0
        if self.parameter_share:
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt', map_location=torch.device(self.device))
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt', map_location=torch.device(self.device))
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)
            policy_cost_critic_state_dict = torch.load(str(self.model_dir) + '/cost_critic_agent' + str(agent_id) + '.pt', map_location=torch.device(self.device))
            self.policy[agent_id].cost_critic.load_state_dict(policy_cost_critic_state_dict)
        else:
            for agent_id in range(self.num_agents):
                policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt', map_location=torch.device(self.device))
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt', map_location=torch.device(self.device))
                self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)
                policy_cost_critic_state_dict = torch.load(str(self.model_dir) + '/cost_critic_agent' + str(agent_id) + '.pt', map_location=torch.device(self.device))
                self.policy[agent_id].cost_critic.load_state_dict(policy_cost_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
