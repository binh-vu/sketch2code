import random, math
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_

from s1x_supervise_baseline import make_toy_vocab_v1, make_dataset_v1, get_toy_dataset_v1, iter_batch
from sketch2code.methods.lstm import LSTMNoEmbedding
from sketch2code.helpers import Placeholder
from sketch2code.methods.dqn import Transition, ReplayMemory
from sketch2code.methods.lstm import prepare_batch_sentences
from sketch2code.methods.rl_env import *
from sketch2code.methods.dqn import conv2d_size_out, pool2d_size_out


class EncoderV1(nn.Module):
    
    def __init__(self, img_h: int, img_w: int, img_repr_size: int):
        super().__init__()
        self.img_w = img_w
        self.img_h = img_h
        self.img_repr_size = img_repr_size

        self.__build_model()

    def __build_model(self):
        # network compute features of target image
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=1)
        self.bn1 = nn.BatchNorm2d(16, momentum=0.9)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(32, momentum=0.9)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(64, momentum=0.9)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        imgsize = [self.img_h, self.img_w]
        for i, s in enumerate(imgsize):
            s = conv2d_size_out(s, 7, 1)
            s = pool2d_size_out(s, 3, 2)
            s = conv2d_size_out(s, 5, 1)
            s = pool2d_size_out(s, 3, 2)
            s = conv2d_size_out(s, 5, 1)
            s = pool2d_size_out(s, 3, 2)
            imgsize[i] = s
        
        linear_input_size = imgsize[0] * imgsize[1] * 64

        self.fc1 = nn.Linear(linear_input_size, self.img_repr_size)
        
    def forward(self, x1):
        """
        :param x1: desired iamges (batch: N x C x H x W)
        :return:
        """
        x1 = self.pool1(F.selu(self.bn1(self.conv1(x1))))
        x1 = self.pool2(F.selu(self.bn2(self.conv2(x1))))
        x1 = self.pool3(F.selu(self.bn3(self.conv3(x1))))

        # flatten to N x (C * W * H)
        x1 = x1.view(x1.shape[0], -1)
        x1 = F.relu(self.fc1(x1))

        return x1


class ActionDecoder(nn.Module):
    
    def __init__(self, img_repr_size: int, dsl_vocab, dsl_hidden_dim, dsl_embedding_dim, n_actions, padding_idx: int=0):
        super().__init__()
        self.img_repr_size = img_repr_size
        self.dsl_vocab = dsl_vocab
        self.dsl_hidden_dim = dsl_hidden_dim
        self.dsl_embedding_dim = dsl_embedding_dim
        self.n_actions = n_actions
        self.padding_idx = padding_idx

        self.__build_model()

    def __build_model(self):
        self.word_embedding = nn.Embedding(
            num_embeddings=len(self.dsl_vocab), embedding_dim=self.dsl_embedding_dim, padding_idx=self.padding_idx)
        self.lstm = LSTMNoEmbedding(
            input_size=self.dsl_embedding_dim + self.img_repr_size,
            hidden_size=self.dsl_hidden_dim,
            n_layers=1)
        self.lstm2action = nn.Linear(self.dsl_hidden_dim, self.n_actions)

    def forward(self, x1, x2, x2_lens):
        """
        :param x1: output from encoder program (batch: N x E)
        :param x2: current programs (batch: N x T)
        :param x2_lens: lengths of current programs (N)
        :return:
        """
        batch_size = x2.shape[0]

        x2 = self.word_embedding(x2)
        x1 = x1.view(batch_size, 1, self.img_repr_size).expand(batch_size, x2.shape[1], x1.shape[1])
        x2 = torch.cat([x1, x2], dim=2)
        x2, (hn, cn) = self.lstm(x2, x2_lens, self.lstm.init_hidden(x2, batch_size))
        # flatten from N x 1 x H to N x (1 * H)
        hn = hn.view(batch_size, -1)
        nt = F.log_softmax(self.lstm2action(hn), dim=1)
        return nt

class DQN(nn.Module):
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x1, x2, x2lens):
        x1 = self.encoder(x1)
        return self.decoder(x1, x2, x2lens)
    
    
class RewardFunc():
    
    def __init__(self, target_gui: np.ndarray):
        self.H, self.W, self.C = target_gui.shape
        self.WC = self.W * self.C
        self.end_content_row = self.H
        for i in range(self.H - 1, -1, -1):
            if target_gui[i].sum() > 0:
                self.end_content_row = i + 1  # exclusive
                break
        self.target_gui = target_gui
        self.max_reward = self.end_content_row * self.W
                
    def reward(self, current_gui: np.ndarray):
        credits = 0
        max_scanned_col = self.W
        
        # step 1: figure out the end-content-row of current gui, we can safely start from the end-content-row of target_gui
        curr_end_content_row = 0
        for i in range(self.end_content_row - 1, -1, -1):
            if current_gui[i].sum() > 0:
                curr_end_content_row = i + 1  # exclusive
        
        for i in range(curr_end_content_row):
            if (self.target_gui[i] == current_gui[i]).sum() == self.WC:
                max_scanned_col = self.W
            else:
                new_max_scanned_col = 0
                for j in range(max_scanned_col):
                    if (self.target_gui[i, j, :] == current_gui[i, j, :]).sum() == 3:
                        new_max_scanned_col += 1
                max_scanned_col = new_max_scanned_col
            credits += max_scanned_col

        # the rest from target_gui empty content is empty space, any match here will be deducted to your credit
        neg_credits = (self.target_gui[self.end_content_row:] != current_gui[self.end_content_row:]).sum() / self.C
        return credits - neg_credits
    

# tag is about to open => its parent
open_tag_penalties = {"div": {"<button>"}, "button": {"<button>"}}
disjoint_classes_penalties = {
    "<div>": {"row", "col-12", "col-6", "col-4", "col-3", "container-fluid", "grey-background"},
    "<button>": {"btn-danger", "btn-warning", "btn-success"},
}    
    
def compute_semantic_penalty(obs: Observation, action: Action, max_penalty: float):
    global open_tag_penalties, disjoint_classes_penalties
    curr_tag = obs.curr_tag
    if action.action_type == AddOpenTagAction.action_type:
        action: AddOpenTagAction
        if len(curr_tag.opening_tags) > 0 and curr_tag.tokens[curr_tag.opening_tags[-1]][0] in open_tag_penalties[
                action.tag_name]:
            return max_penalty

        return 0
    elif action.action_type == AddOpenTagAndClassAction.action_type:
        action: AddOpenTagAndClassAction
        if len(curr_tag.opening_tags) > 0:
            prev_token, prev_classes = curr_tag.tokens[curr_tag.opening_tags[-1]]

            if prev_token in open_tag_penalties[action.tag]:
                return max_penalty

            if prev_token == "<div>" and action.tag == "div":
                # layout constraint

                # should not have nested container
                violate_layout_constraint = action.classes[0].startswith("container-fluid")
                violate_layout_constraint = violate_layout_constraint or (action.classes[0].startswith("col-")
                                                                          and not prev_classes[0].startswith("row"))
                violate_layout_constraint = violate_layout_constraint or (action.classes[0].startswith("row") and
                                                                          not prev_classes[0].startswith("container"))
                # should not have duplicated class
                violate_layout_constraint = violate_layout_constraint or (action.classes[0] == prev_classes[0])
                # row follows by col
                violate_layout_constraint = violate_layout_constraint or (not action.classes[0].startswith('col-')
                                                                          and prev_classes[0].startswith("row"))

                if violate_layout_constraint:
                    return max_penalty
        else:
            if not (action.tag == "div" and action.classes[0].startswith("container")):
                # top level should be container
                return max_penalty

        return 0
    elif action.action_type == AddClassAction.action_type:
        action: AddClassAction
        # invalid action has been filter out
        token, classes = curr_tag.tokens[curr_tag.opening_tags[-1]]
        disjoint_classes = disjoint_classes_penalties[token]
        if action.cls in disjoint_classes and any(c in disjoint_classes for c in classes):
            return max_penalty

        return 0
    elif action.action_type == AddCloseTagAction.action_type:
        return 0

    raise Exception("Invalid action", action)

def select_action(policy_net,
                  env_creator: EnvCreator,
                  obs: Observation,
                  images,
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
                images[obs.env_idx:obs.env_idx+1].to(device), dsl_tokens,
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
                   images: torch.tensor,
                   g: Placeholder,
                   device=None,
                   clip_grad_val=5.):
    if len(memory) < g.BATCH_SIZE:
        return

    transitions = memory.sample(g.BATCH_SIZE)
    # create a batch of state, batch of action, batch of next state and batch of reward
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(batch.not_done, device=device, dtype=torch.uint8)

    state_x2, state_x2len, sorted_idx = prepare_batch_sentences([env_creator.tag2dsl(obs.curr_tag) for obs in batch.state],
                                                device=device)
    state_x1 = images[[batch.state[i].env_idx for i in sorted_idx]].to(device)

    non_final_next_states = [s for d, s in zip(batch.not_done, batch.next_state) if d]
    non_final_nstate_x2, non_final_nstate_x2len, sorted_idx = prepare_batch_sentences(
        [env_creator.tag2dsl(obs.curr_tag) for obs in non_final_next_states], device=device)
    non_final_nstate_x1 = images[[non_final_next_states[i].env_idx for i in sorted_idx]].to(device)

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
    clip_grad_value_(policy_net.parameters(), clip_grad_val)
    optimizer.step()
    
    return float(loss)
