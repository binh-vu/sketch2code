#!/usr/bin/python
# -*- coding: utf-8 -*-
import copy

import numpy as np
from typing import *

import torch

from sketch2code.data_model import Tag, LinearizedTag
from sketch2code.helpers import norm_rgb_imgs, shrink_img, viz_grid
from sketch2code.render_engine import RemoteRenderEngine


class Action:

    action_type = ""

    def __init__(self, action_id: int):
        self.action_id = action_id

    def exec(self, state: 'Observation') -> Optional[LinearizedTag]:
        raise NotImplementedError()

    def is_valid(self, obs: 'Observation') -> bool:
        raise NotImplementedError()


class EnvCreator:
    def __init__(self, tags: List[Tag], sketches: List[np.ndarray], shrink_factor: float, device=None):
        self.render_engine: RemoteRenderEngine = RemoteRenderEngine.get_instance(
            tags[0].to_html(), sketches[0].shape[1], sketches[0].shape[0])

        self.actions: List[Action] = [AddCloseTagAction(0)]
        for tag, classes in tags[0].supported_tags.items():
            if tag == 'html':
                continue

            self.actions.append(AddOpenTagAction(len(self.actions), tag))
            for cls in classes:
                self.actions.append(AddClassAction(len(self.actions), tag, cls))

        self.vocab = self.__build_dsl_vocab(tags[0].supported_tags)
        self.tags = tags
        self.shrink_factor = shrink_factor
        self.device = device
        assert shrink_factor <= 1
        # make sketches in N x C x H x W
        if shrink_factor != 1:
            self.sketches = [shrink_img(img, shrink_factor).transpose((2, 0, 1)) for img in sketches]
        else:
            self.sketches = [s.transpose((2, 0, 1)) for s in sketches]

    def create(self):
        return [Env(self, tag, sketch) for tag, sketch in zip(self.tags, self.sketches)]

    def __build_dsl_vocab(self, supported_tags):
        tokens = []
        for tag, classes in supported_tags.items():
            tokens.append(f"<{tag}>")
            tokens.append(f"</{tag}>")
            for cls in classes:
                tokens.append(f"cls={cls}")
        tokens.sort()
        # add start so that input to lstm is not empty
        vocab = {'<pad>': 0, '<start>': 1}
        for i, token in enumerate(tokens, start=2):
            vocab[token] = i
        return vocab

    def tag2dsl(self, tag: LinearizedTag):
        dsl_tokens = [self.vocab['<start>']]
        for token, classes in tag.tokens:
            dsl_tokens.append(self.vocab[token])
            for cls in classes:
                dsl_tokens.append(self.vocab[f'cls={cls}'])
        return dsl_tokens

    # def render_imgs(self, tags: List[LinearizedTag]):
    #     imgs = self.render_engine.render_pages(tags)
    #     imgs = norm_rgb_imgs(imgs)
    #
    #     return imgs

    def render_img(self, tag: LinearizedTag):
        img = self.render_engine.render_page(tag)
        img = norm_rgb_imgs(img, dtype=np.float32)
        if self.shrink_factor != 1:
            img = shrink_img(img, self.shrink_factor)

        return img.transpose((2, 0, 1))


class Observation:
    def __init__(self, tensor_goal_state: torch.tensor, goal_state: np.ndarray, goal_tag: Tag, curr_state: np.ndarray,
                 curr_tag: LinearizedTag):
        self.tensor_goal_state = tensor_goal_state
        self.goal_tag = goal_tag
        self.goal_state = goal_state
        self.curr_tag = curr_tag
        self.curr_state = curr_state


class Env:
    def __init__(self, env_creator: EnvCreator, goal_tag: Tag, sketch: np.ndarray):
        self.env_creator = env_creator
        self.goal_tag = goal_tag

        self.obs = Observation(
            torch.tensor(sketch, device=env_creator.device), sketch, goal_tag, np.zeros_like(sketch),
            LinearizedTag.default())
        self.matching_score = compute_matching_score(self.obs.goal_state, self.obs.curr_state)
        self.max_matching_score = np.prod(sketch.shape)

        self.semantic_penalty = -5000

    def render(self):
        return viz_grid(
            np.stack([self.obs.goal_state, self.obs.curr_state], axis=0).transpose((0, 2, 3, 1)), padding_color=1)

    def reset(self):
        self.obs = Observation(self.obs.tensor_goal_state, self.obs.goal_state, self.goal_tag,
                               np.zeros_like(self.obs.curr_state), LinearizedTag.default())
        self.matching_score = compute_matching_score(self.obs.goal_state, self.obs.curr_state)
        self.max_matching_score = np.prod(self.obs.goal_state.shape)

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        new_tag = action.exec(self.obs)

        if new_tag is None:
            # penalty
            return self.obs, self.semantic_penalty, False, {}

        new_sketch = self.env_creator.render_img(new_tag)
        new_obs = Observation(self.obs.tensor_goal_state, self.obs.goal_state, self.obs.goal_tag, new_sketch, new_tag)
        new_score = compute_matching_score(self.obs.goal_state, new_sketch)

        # compute the reward
        if isinstance(action, AddCloseTagAction):
            reward = 0
            done = new_tag.is_valid() and new_score == self.max_matching_score
        else:
            reward = (new_score - self.matching_score)
            done = isinstance(action, AddClassAction) and new_score == self.max_matching_score

        semantic_penalty = compute_semantic_penalty(self.obs, action, self.semantic_penalty)
        self.obs = new_obs
        self.matching_score = new_score

        return new_obs, reward + semantic_penalty, done, {}


def compute_matching_score(goal_state: np.ndarray, curr_state: np.ndarray) -> float:
    return (goal_state == curr_state).sum()


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

    def exec(self, obs: Observation) -> bool:
        tag = obs.curr_tag.clone()
        tag.add_tag_and_class(self.tag, self.classes)
        return tag

    def __repr__(self):
        return f'AddOpenTagAndClass(<{self.tag} class="{" ".join(self.classes)}">)'
