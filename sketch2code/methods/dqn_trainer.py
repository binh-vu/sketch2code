#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import random

import torch
import torch.nn.functional as F
from sketch2code.helpers import Placeholder
from sketch2code.methods.dqn import Transition, ReplayMemory
from sketch2code.methods.lstm import prepare_batch_sents
from sketch2code.methods.rl_env import Observation, EnvCreator, Env


def select_action(policy_net,
                  env_creator: EnvCreator,
                  obs: Observation,
                  g: Placeholder,
                  device=None,
                  eps_threshold: float = None):
    sample = random.random()
    if eps_threshold is None:
        eps_threshold = g.EPS_END + (g.EPS_START - g.EPS_END) * math.exp(-1 * g.step_done / g.EPS_DECAY)
        g.step_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            dsl_tokens = torch.tensor(env_creator.tag2dsl(obs.curr_tag), dtype=torch.long, device=device).view(1, -1)
            dsl_tokens_lens = torch.tensor([dsl_tokens.shape[1]], dtype=torch.long, device=device)

            return env_creator.actions[policy_net(
                obs.tensor_goal_state.view(1, *obs.tensor_goal_state.shape), dsl_tokens,
                dsl_tokens_lens).max(1)[1].view(1, 1)]
            # qval = policy_net(obs.tensor_goal_state.view(1, *obs.tensor_goal_state.shape), dsl_tokens, dsl_tokens_lens)[0]
            #
            # for action_idx in qval.argsort():
            #     if env_creator.actions[action_idx].is_valid(obs):
            #         return env_creator.actions[action_idx]
    else:
        return env_creator.actions[random.randrange(g.n_actions)]
        # actions = list(range(g.n_actions))
        # random.shuffle(actions)
        # for action_idx in actions:
        #     if env_creator.actions[action_idx].is_valid(obs):
        #         return env_creator.actions[action_idx]

    raise Exception("[BUG] No valid action found!")


def optimize_model(policy_net,
                   target_net,
                   optimizer,
                   env_creator: EnvCreator,
                   memory: ReplayMemory,
                   g: Placeholder,
                   device=None):
    if len(memory) < g.BATCH_SIZE:
        return

    transitions = memory.sample(g.BATCH_SIZE)
    # create a batch of state, batch of action, batch of next state and batch of reward
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(batch.not_done, device=device, dtype=torch.uint8)

    state_x1 = torch.stack([obs.tensor_goal_state for obs in batch.state])
    state_x2, state_x2len = prepare_batch_sents([env_creator.tag2dsl(obs.curr_tag) for obs in batch.state],
                                                device=device)

    non_final_next_states = [s for d, s in zip(batch.not_done, batch.next_state) if d]
    non_final_nstate_x1 = torch.stack([obs.tensor_goal_state for obs in non_final_next_states])
    non_final_nstate_x2, non_final_nstate_x2len = prepare_batch_sents(
        [env_creator.tag2dsl(obs.curr_tag) for obs in non_final_next_states], device=device)

    action_batch = torch.tensor(batch.action, device=device).view(-1, 1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_x1, state_x2, state_x2len).gather(1, action_batch)  # N x 1

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(g.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_nstate_x1, non_final_nstate_x2,
                                                   non_final_nstate_x2len).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * g.GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
