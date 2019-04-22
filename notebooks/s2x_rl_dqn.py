#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import random
from collections import namedtuple
from typing import *

import cv2
import numpy as np
import torch

from sketch2code.data_model import Tag, LinearizedTag
from sketch2code.helpers import norm_rgb_imgs, shrink_img, viz_grid, Placeholder
from sketch2code.synthesize_program import HTMLProgram


class Action:
    action_type = ""

    def __init__(self, action_id: int):
        self.action_id = action_id

    def exec(self, state: 'Observation') -> Optional[LinearizedTag]:
        raise NotImplementedError()

    def is_valid(self, obs: 'Observation') -> bool:
        raise NotImplementedError()


class Observation:
    def __init__(self, env_idx: int, img: np.ndarray, tag: LinearizedTag, feedback=None):
        self.env_idx = env_idx
        self.tag = tag
        self.img = img
        self.feedback = feedback  # to store feedback from teacher


class AddOpenTagAction(Action):
    action_type = "add_open_tag"

    def __init__(self, action_id: int, tag_name: str):
        super().__init__(action_id)
        self.tag_name = tag_name

    def is_valid(self, obs: Observation) -> bool:
        return True

    def exec(self, state: Observation) -> Optional[LinearizedTag]:
        tag = state.tag.clone()
        tag.add_open_tag(self.tag_name)
        return tag

    def __repr__(self):
        return f"AddOpenTag(<{self.tag_name}>)"


class AddCloseTagAction(Action):
    action_type = "add_close_tag"

    def is_valid(self, obs: Observation) -> bool:
        return obs.tag.can_add_close_tag()

    def exec(self, state: Observation) -> Optional[LinearizedTag]:
        tag = state.tag.clone()
        if not tag.add_close_tag():
            return None
        return tag

    def __repr__(self):
        return f"AddCloseTag()"


class AddClassAction(Action):
    action_type = "add_class"

    def __init__(self, action_id: int, tag: str, cls: str):
        super().__init__(action_id)
        self.tag = tag
        self.cls = cls

    def is_valid(self, obs: Observation) -> bool:
        return obs.tag.can_add_class(self.tag, self.cls)

    def exec(self, state: Observation) -> Optional[LinearizedTag]:
        tag = state.tag.clone()
        if not tag.add_class(self.tag, self.cls):
            return None
        return tag

    def __repr__(self):
        return f"AddClass(<{self.tag} class={self.cls}>)"


class AddOpenTagAndClassAction(Action):
    action_type = "add_open_tag_and_class"

    def __init__(self, action_id: int, tag: str, classes: Tuple[str, ...]):
        super().__init__(action_id)
        self.tag = tag
        self.classes = classes

    def is_valid(self, obs: Observation) -> bool:
        return True

    def exec(self, obs: Observation) -> Optional[LinearizedTag]:
        tag = obs.tag.clone()
        tag.add_tag_and_class(self.tag, self.classes)
        return tag

    def __repr__(self):
        return f'AddOpenTagAndClass(<{self.tag} class="{" ".join(self.classes)}">)'


class UndoAction(Action):
    action_type = "undo"

    def __init__(self, action_id: int):
        super().__init__(action_id)

    def is_valid(self, obs: Observation) -> bool:
        return len(obs.tag.str_tokens) > 0

    def exec(self, obs: Observation) -> Optional[LinearizedTag]:
        if not self.is_valid(obs):
            return None
        tag = obs.tag.clone()
        tag.pop()
        return tag

    def __repr__(self):
        return f'Undo()'


class EnvCreator:
    def __init__(self,
                 render_engine,
                 tags: List[Tag],
                 vocab: Dict[str, int],
                 sketches: List[np.ndarray],
                 shrink_factor: float,
                 interpolation=cv2.INTER_NEAREST,
                 device=None):
        self.render_engine = render_engine

        self.actions: List[Action] = [UndoAction(0), AddCloseTagAction(1)]
        for w, i in vocab.items():
            if w not in {"<pad>", "<program>", "</program>"}:
                tag, tag_type, classes = HTMLProgram.token2tag(w)
                if tag_type == HTMLProgram.OPEN_TAG:
                    self.actions.append(AddOpenTagAndClassAction(len(self.actions), tag, classes))
                elif tag_type == HTMLProgram.SPECIAL_TOKEN:
                    raise NotImplemented()

        self.vocab = vocab
        self.tags = tags
        self.shrink_factor = shrink_factor
        self.interpolation = interpolation
        self.device = device
        self.sketches = sketches
        assert shrink_factor <= 1

    def create(self):
        return [Env(self, idx) for idx in range(len(self.tags))]

    def tag2dsl(self, tag: LinearizedTag):
        dsl_tokens = [self.vocab['<program>']]
        for token in tag.str_tokens:
            dsl_tokens.append(self.vocab[token])
        return dsl_tokens

    def render_img(self, tag: LinearizedTag):
        img = self.render_engine.render_page(tag)
        img = norm_rgb_imgs(img, dtype=np.float32)
        img = shrink_img(img, self.shrink_factor, self.interpolation)
        return img


TeacherReward = namedtuple("TeacherReward", ["reward", "progress"])


class Teacher:

    def reward4ignorance(self) -> float:
        """Return a reward if student make an obvious wrong action"""
        raise NotImplementedError()

    def reward(self, prev_obs: Observation, obs: Observation, action: Action) -> TeacherReward:
        """Compute a reward based on the current observation"""
        raise NotImplementedError()

    def should_retry(self, obs: Observation) -> bool:
        """Telling if student would need to retry the lesson"""
        raise NotImplementedError()


class Env:
    def __init__(self, env_creator: EnvCreator, env_idx: int):
        self.env_creator = env_creator
        self.env_idx = env_idx

        self.obs = Observation(env_idx, np.zeros_like(env_creator.sketches[env_idx]), LinearizedTag.default())
        self.teacher: Teacher = None

    def set_sponsor(self, teacher: Teacher):
        self.teacher = teacher

    def get_target_state(self):
        return self.env_creator.sketches[self.env_idx]

    def render(self, obs=None):
        return viz_grid(
            np.stack([self.get_target_state(), (self.obs if obs is None else obs).img], axis=0), padding_color=1)

    def reset(self):
        self.obs = Observation(self.env_idx, np.zeros_like(self.get_target_state()), LinearizedTag.default())

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        new_tag = action.exec(self.obs)
        if new_tag is None:
            # penalty
            return self.obs, self.teacher.reward4ignorance(), False, {}

        new_sketch = self.env_creator.render_img(new_tag)
        next_obs = Observation(self.env_idx, new_sketch, new_tag)
        teacher_feedback = self.teacher.reward(self.obs, next_obs, action)
        self.obs = next_obs
        return self.obs, teacher_feedback.reward, teacher_feedback.progress == 1.0, {}


Transition = namedtuple('Transition', ('env_idx', 'curr_state', 'next_state', 'action', 'abs_reward', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.total_abs_reward = 0.0
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition: Transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        if self.memory[self.position] is not None:
            self.total_abs_reward = self.total_abs_reward - self.memory[
                self.position].abs_reward + transition.abs_reward
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if self.total_abs_reward == 0.0:
            return random.sample(self.memory, batch_size)
        return np.random.choice(
            range(len(self.memory)),
            batch_size,
            replace=True,
            p=[x.abs_reward / self.total_abs_reward for x in self.memory])

    def __len__(self):
        return len(self.memory)


def select_action(policy_q,
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
            dsl_tokens = torch.tensor(env_creator.tag2dsl(obs.tag), dtype=torch.long, device=device).view(1, -1)
            dsl_tokens_lens = torch.tensor([dsl_tokens.shape[1]], dtype=torch.long, device=device)

            return env_creator.actions[policy_q(images[obs.env_idx:obs.env_idx + 1].to(device), dsl_tokens,
                                                dsl_tokens_lens).max(1)[1].view(1, 1)]
    else:
        return env_creator.actions[random.randrange(g.n_actions)]
