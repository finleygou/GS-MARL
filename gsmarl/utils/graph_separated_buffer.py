import torch
import numpy as np
from collections import defaultdict
from gsmarl.utils.util import check, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1,0,2).reshape(-1, *x.shape[2:])


class GraphSeparatedReplayBuffer(object):
    def __init__(self, args, graph_obs_shape, share_graph_obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.rnn_hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.algo = args.algorithm_name

        nodes_feats_shape, edge_index_shape, edge_attr_shape = graph_obs_shape
        share_nodes_feats_shape, share_edge_index_shape, share_edge_attr_shape = share_graph_obs_space

        self.aver_episode_costs = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)

        self.nodes_feats = np.zeros((self.episode_length + 1, self.n_rollout_threads, *nodes_feats_shape), dtype=np.float32)
        self.edge_index = np.zeros((self.episode_length + 1, self.n_rollout_threads, *edge_index_shape), dtype=np.float32)
        self.edge_attr = np.zeros((self.episode_length + 1, self.n_rollout_threads, *edge_attr_shape), dtype=np.float32)

        self.share_nodes_feats = np.zeros((self.episode_length + 1, self.n_rollout_threads, *share_nodes_feats_shape), dtype=np.float32)
        self.share_edge_index = np.zeros((self.episode_length + 1, self.n_rollout_threads, *share_edge_index_shape), dtype=np.float32)
        self.share_edge_attr = np.zeros((self.episode_length + 1, self.n_rollout_threads, *share_edge_attr_shape), dtype=np.float32)

        self.rnn_states = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)
        self.rnn_states_cost = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.returns = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        
        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, act_space.n), dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros((self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros((self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        self.agent_id = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.int32)

        self.costs = np.zeros_like(self.rewards)
        self.cost_preds = np.zeros_like(self.value_preds)
        self.cost_returns = np.zeros_like(self.returns)
        
        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.factor = None

        self.step = 0

    def update_factor(self, factor):
        self.factor = factor.copy()

    def return_aver_insert(self, aver_episode_costs):
        self.aver_episode_costs = aver_episode_costs.copy()

    def insert(self, agent_id, nodes_feats, edge_index, edge_attr, share_nodes_feats, share_edge_index, share_edge_attr, rnn_states, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None, costs=None,
               cost_preds=None, rnn_states_cost=None, done_episodes_costs_aver=None, aver_episode_costs = 0):

        self.nodes_feats[self.step + 1] = nodes_feats.copy()
        self.edge_index[self.step + 1] = edge_index.copy()
        self.edge_attr[self.step + 1] = edge_attr.copy()
        self.share_nodes_feats[self.step + 1] = share_nodes_feats.copy()
        self.share_edge_index[self.step + 1] = share_edge_index.copy()
        self.share_edge_attr[self.step + 1] = share_edge_attr.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.agent_id[self.step + 1] = agent_id.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()
        if costs is not None:
            self.costs[self.step] = costs.copy()
        if cost_preds is not None:
            self.cost_preds[self.step] = cost_preds.copy()
        if rnn_states_cost is not None:
            self.rnn_states_cost[self.step + 1] = rnn_states_cost.copy()

        self.step = (self.step + 1) % self.episode_length

    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length
    
    def after_update(self):
        self.agent_id[0] = self.agent_id[-1].copy()
        self.share_nodes_feats[0] = self.share_nodes_feats[-1].copy()
        self.share_edge_index[0] = self.share_edge_index[-1].copy()
        self.share_edge_attr[0] = self.share_edge_attr[-1].copy()
        self.nodes_feats[0] = self.nodes_feats[-1].copy()
        self.edge_index[0] = self.edge_index[-1].copy()
        self.edge_attr[0] = self.edge_attr[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.rnn_states_cost[0] = self.rnn_states_cost[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def chooseafter_update(self):
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """
        use proper time limits, the difference of use or not is whether use bad_mask
        """
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[
                            step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                            + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                            + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def compute_cost_returns(self, next_cost, value_normalizer=None):

        if self._use_proper_time_limits:
            if self._use_gae:
                self.cost_preds[-1] = next_cost
                gae = 0
                for step in reversed(range(self.costs.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.costs[step] + self.gamma * value_normalizer.denormalize(self.cost_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.cost_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.cost_returns[step] = gae + value_normalizer.denormalize(self.cost_preds[step])
                    else:
                        delta = self.costs[step] + self.gamma * self.cost_preds[step + 1] * self.masks[step + 1] - self.cost_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.cost_returns[step] = gae + self.cost_preds[step]
            else:
                self.cost_returns[-1] = next_cost
                for step in reversed(range(self.costs.shape[0])):
                    if self._use_popart:
                        self.cost_returns[step] = (self.cost_returns[step + 1] * self.gamma * self.masks[step + 1] + self.costs[step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(self.cost_preds[step])
                    else:
                        self.cost_returns[step] = (self.cost_returns[step + 1] * self.gamma * self.masks[step + 1] + self.costs[step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * self.cost_preds[step]
        else:
            if self._use_gae:
                self.cost_preds[-1] = next_cost
                gae = 0
                for step in reversed(range(self.costs.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.costs[step] + self.gamma * value_normalizer.denormalize(self.cost_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.cost_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.cost_returns[step] = gae + value_normalizer.denormalize(self.cost_preds[step])
                    else:
                        delta = self.costs[step] + self.gamma * self.cost_preds[step + 1] * self.masks[step + 1] - self.cost_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.cost_returns[step] = gae + self.cost_preds[step]
            else:
                self.cost_returns[-1] = next_cost
                for step in reversed(range(self.costs.shape[0])):
                    self.cost_returns[step] = self.cost_returns[step + 1] * self.gamma * self.masks[step + 1] + self.costs[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None, cost_adv=None):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "".format(n_rollout_threads, episode_length, n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        agent_id = self.agent_id[:-1].reshape(-1, *self.agent_id.shape[2:])
        nodes_feats = self.nodes_feats[:-1].reshape(-1, *self.nodes_feats.shape[2:])
        edge_index = self.edge_index[:-1].reshape(-1, *self.edge_index.shape[2:])
        edge_attr = self.edge_attr[:-1].reshape(-1, *self.edge_attr.shape[2:])
        share_nodes_feats = self.share_nodes_feats[:-1].reshape(-1, *self.share_nodes_feats.shape[2:])
        share_edge_index = self.share_edge_index[:-1].reshape(-1, *self.share_edge_index.shape[2:])
        share_edge_attr = self.share_edge_attr[:-1].reshape(-1, *self.share_edge_attr.shape[2:])

        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        rnn_states_cost = self.rnn_states_cost[:-1].reshape(-1, *self.rnn_states_cost.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        cost_preds = self.cost_preds[:-1].reshape(-1, 1)
        cost_returns = self.cost_returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        # print("self.aver_episode_costs--separated--buffer", self.aver_episode_costs.mean())
        aver_episode_costs = self.aver_episode_costs # self.aver_episode_costs[:-1].reshape(-1, *self.aver_episode_costs.shape[2:])
        if self.factor is not None:
            # factor = self.factor.reshape(-1,1)
            factor = self.factor.reshape(-1, self.factor.shape[-1])
        advantages = advantages.reshape(-1, 1)
        if cost_adv is not None:
            cost_adv = cost_adv.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            agent_id_batch = agent_id[indices]
            nodes_feats_batch = nodes_feats[indices]
            edge_index_batch = edge_index[indices]
            edge_attr_batch = edge_attr[indices]
            share_nodes_feats_batch = share_nodes_feats[indices]
            share_edge_index_batch = share_edge_index[indices]
            share_edge_attr_batch = share_edge_attr[indices]

            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            rnn_states_cost_batch = rnn_states_cost[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            cost_preds_batch = cost_preds[indices]
            cost_return_batch = cost_returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]
            if cost_adv is None:
                cost_adv_targ = None
            else:
                cost_adv_targ = cost_adv[indices]

            if self.factor is None:
                yield nodes_feats_batch, edge_index_batch, edge_attr_batch, share_nodes_feats_batch, share_edge_index_batch, share_edge_attr_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch
            else:
                factor_batch = factor[indices]
                yield agent_id_batch, share_nodes_feats_batch, share_edge_index_batch, share_edge_attr_batch, nodes_feats_batch, edge_index_batch, edge_attr_batch,  rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch, cost_preds_batch, cost_return_batch, rnn_states_cost_batch, cost_adv_targ, aver_episode_costs

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length, cost_adv=None):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length  # [C=r*T/L]
        mini_batch_size = data_chunks // num_mini_batch
    
        assert episode_length * n_rollout_threads >= data_chunk_length, (
            "requires the number of processes ({}) * episode length ({}) "
            "to be greater than or equal to the number of " 
            "data chunk length ({}).".format(n_rollout_threads, episode_length, data_chunk_length))
        assert episode_length % data_chunk_length == 0, (
            "requires episode length ({}) mod data_chunk_length ({}) == 0 ".format(episode_length, data_chunk_length))
        assert data_chunks >= 2, ("need larger batch size")
    
        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]
    
        if len(self.share_nodes_feats.shape) > 3:
            share_nodes_feats = self.share_nodes_feats[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.share_nodes_feats.shape[2:])
            share_edge_index = self.share_edge_index[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.share_edge_index.shape[2:])
            share_edge_attr = self.share_edge_attr[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.share_edge_attr.shape[2:])
            nodes_feats = self.nodes_feats[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.nodes_feats.shape[2:])
            edge_index = self.edge_index[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.edge_index.shape[2:])
            edge_attr = self.edge_attr[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.edge_attr.shape[2:])
            agent_id = self.agent_id[:-1].transpose(1, 0, 2).reshape(-1, *self.agent_id.shape[2:])
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])
    
        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        if cost_adv is None:
            cost_advantages = None
        else:
            cost_advantages = _cast(cost_adv)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        if self.factor is not None:
            factor = _cast(self.factor)
        cost_preds = _cast(self.cost_preds[:-1])
        cost_return = _cast(self.cost_returns[:-1])

        rnn_states = self.rnn_states[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states_critic.shape[2:])
        rnn_states_cost = self.rnn_states_cost[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states_cost.shape[2:])
    
        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        aver_episode_costs = self.aver_episode_costs

        for indices in sampler:

            share_nodes_feats_batch = []
            share_edge_index_batch = []
            share_edge_attr_batch = []
            nodes_feats_batch = []
            edge_index_batch = []
            edge_attr_batch = []
            agent_id_batch = []

            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            factor_batch = []
            cost_preds_batch = []
            cost_return_batch = []
            rnn_states_cost_batch = []
            cost_adv_targ = []

            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N Dim]-->[N T Dim]-->[T*N,Dim]-->[L,Dim]
                share_nodes_feats_batch.append(share_nodes_feats[ind:ind+data_chunk_length])
                share_edge_index_batch.append(share_edge_index[ind:ind+data_chunk_length])
                share_edge_attr_batch.append(share_edge_attr[ind:ind+data_chunk_length])
                nodes_feats_batch.append(nodes_feats[ind:ind+data_chunk_length])
                edge_index_batch.append(edge_index[ind:ind+data_chunk_length])
                edge_attr_batch.append(edge_attr[ind:ind+data_chunk_length])
                agent_id_batch.append(agent_id[ind:ind+data_chunk_length])

                actions_batch.append(actions[ind:ind+data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind+data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind+data_chunk_length])
                return_batch.append(returns[ind:ind+data_chunk_length])
                masks_batch.append(masks[ind:ind+data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind+data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind+data_chunk_length])
                adv_targ.append(advantages[ind:ind+data_chunk_length])
                if cost_adv is not None:
                    cost_adv_targ.append(cost_advantages[ind:ind+data_chunk_length])
                # size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_cost_batch.append(rnn_states_cost[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])
                if self.factor is not None:
                    factor_batch.append(factor[ind:ind+data_chunk_length])
                cost_preds_batch.append(cost_preds[ind:ind+data_chunk_length])
                cost_return_batch.append(cost_return[ind:ind+data_chunk_length])

            L, N = data_chunk_length, mini_batch_size
    
            # These are all from_numpys of size (N, L, Dim)
            share_nodes_feats_batch = np.stack(share_nodes_feats_batch)
            share_edge_index_batch = np.stack(share_edge_index_batch)
            share_edge_attr_batch = np.stack(share_edge_attr_batch)
            agent_id_batch = np.stack(agent_id_batch)
            nodes_feats_batch = np.stack(nodes_feats_batch)
            edge_index_batch = np.stack(edge_index_batch)
            edge_attr_batch = np.stack(edge_attr_batch)
    
            actions_batch = np.stack(actions_batch)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch)
            if self.factor is not None:
                factor_batch = np.stack(factor_batch)
            value_preds_batch = np.stack(value_preds_batch)
            return_batch = np.stack(return_batch)
            masks_batch = np.stack(masks_batch)
            active_masks_batch = np.stack(active_masks_batch)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch)
            adv_targ = np.stack(adv_targ)
            cost_adv_targ = np.stack(cost_adv_targ)
            cost_preds_batch = np.stack(cost_preds_batch)
            cost_return_batch = np.stack(cost_return_batch)
    
            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[2:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[2:])
            rnn_states_cost_batch = np.stack(rnn_states_cost_batch).reshape(N, *self.rnn_states_cost.shape[2:])
    
            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_nodes_feats_batch = _flatten(L, N, share_nodes_feats_batch)
            share_edge_index_batch = _flatten(L, N, share_edge_index_batch)
            share_edge_attr_batch = _flatten(L, N, share_edge_attr_batch)
            nodes_feats_batch = _flatten(L, N, nodes_feats_batch)
            edge_index_batch = _flatten(L, N, edge_index_batch)
            edge_attr_batch = _flatten(L, N, edge_attr_batch)
            agent_id_batch = _flatten(L, N, agent_id_batch)

            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            if self.factor is not None:
                factor_batch = _flatten(L, N, factor_batch)
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            cost_preds_batch = _flatten(L, N, cost_preds_batch)
            cost_return_batch = _flatten(L, N, cost_return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)
            cost_adv_targ = _flatten(L, N, cost_adv_targ)

            yield agent_id_batch, share_nodes_feats_batch, share_edge_index_batch, share_edge_attr_batch, nodes_feats_batch, edge_index_batch, edge_attr_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch, cost_preds_batch, cost_return_batch, rnn_states_cost_batch, cost_adv_targ, aver_episode_costs         