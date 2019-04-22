#!/usr/bin/python
# -*- coding: utf-8 -*-
import copy, cv2

import numpy as np
from typing import *

import torch

from sketch2code.data_model import Tag, LinearizedTag
from sketch2code.helpers import norm_rgb_imgs, shrink_img, viz_grid
from sketch2code.render_engine import RemoteRenderEngine
from sketch2code.synthesize_program import HTMLProgram

class Action:

    action_type = ""

    def __init__(self, action_id: int):
        self.action_id = action_id

    def exec(self, state: 'Observation') -> Optional[LinearizedTag]:
        raise NotImplementedError()

    def is_valid(self, obs: 'Observation') -> bool:
        raise NotImplementedError()


class EnvCreator:
    def __init__(self, render_engine, tags: List[Tag], vocab: Dict[str, int], sketches: List[np.ndarray], shrink_factor: float, interpolation=cv2.INTER_NEAREST, device=None):
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


class Observation:
    def __init__(self, env_idx: int, curr_state: np.ndarray, curr_tag: LinearizedTag):
        self.env_idx = env_idx
        self.curr_tag = curr_tag
        self.curr_state = curr_state


class Env:
    def __init__(self, env_creator: EnvCreator, env_idx: int):
        self.env_creator = env_creator
        self.env_idx = env_idx
        
        self.obs = Observation(env_idx, np.zeros_like(env_creator.sketches[env_idx]), LinearizedTag.default())
        self.current_reward = 0
        self.max_reward = None
        self.reward_func = None
        self.invalid_action_penalty = -100
        
    def set_reward_func(self, reward_func, max_reward):
        self.reward_func = reward_func
        self.current_reward = 0
        self.max_reward = None
        
    def get_target_state(self):
        return self.env_creator.sketches[self.env_idx]

    def render(self):
        return viz_grid(
            np.stack([self.get_target_state(), self.obs.curr_state], axis=0), padding_color=1)

    def reset(self):
        self.obs = Observation(self.env_idx, np.zeros_like(self.get_target_state()), LinearizedTag.default())
        self.current_reward = 0

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        new_tag = action.exec(self.obs)
        if new_tag is None:
            # penalty
            return self.obs, -100, False, {}
        
        new_sketch = self.env_creator.render_img(new_tag)
        new_obs = Observation(self.env_idx, new_sketch, new_tag)
        new_reward = self.reward_func(new_sketch)

        # compute the reward
        if isinstance(action, AddCloseTagAction):
            diff_reward = 0
            done = new_tag.is_valid() and new_reward == self.max_reward
        else:
            diff_reward = (new_reward - self.current_reward)
            done = isinstance(action, AddClassAction) and new_score == self.max_reward

        self.obs = new_obs
        self.current_reward = new_reward

        return new_obs, diff_reward, done, {}


class AddOpenTagAction(Action):

    action_type = "add_open_tag"

    def __init__(self, action_id: int, tag_name: str):
        super().__init__(action_id)
        self.tag_name = tag_name

    def is_valid(self, obs: Observation) -> bool:
        return True

    def exec(self, state: Observation) -> Optional[LinearizedTag]:
        tag = state.curr_tag.clone()
        tag.add_open_tag(self.tag_name)
        return tag

    def __repr__(self):
        return f"AddOpenTag(<{self.tag_name}>)"


class AddCloseTagAction(Action):
    action_type = "add_close_tag"

    def is_valid(self, obs: Observation) -> bool:
        return obs.curr_tag.can_add_close_tag()

    def exec(self, state: Observation) -> Optional[LinearizedTag]:
        tag = state.curr_tag.clone()
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
        return obs.curr_tag.can_add_class(self.tag, self.cls)

    def exec(self, state: Observation) -> Optional[LinearizedTag]:
        tag = state.curr_tag.clone()
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
        tag = obs.curr_tag.clone()
        tag.add_tag_and_class(self.tag, self.classes)
        return tag

    def __repr__(self):
        return f'AddOpenTagAndClass(<{self.tag} class="{" ".join(self.classes)}">)'

class UndoAction(Action):
    action_type = "undo"

    def __init__(self, action_id: int):
        super().__init__(action_id)

    def is_valid(self, obs: Observation) -> bool:
        return len(obs.curr_tag.str_tokens) > 0

    def exec(self, obs: Observation) -> Optional[LinearizedTag]:
        if not self.is_valid(obs):
            return None
        tag = obs.curr_tag.clone()
        tag.pop()
        return tag

    def __repr__(self):
        return f'Undo()'
