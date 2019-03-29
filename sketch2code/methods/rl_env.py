#!/usr/bin/python
# -*- coding: utf-8 -*-
import copy

import numpy as np
from typing import *

from sketch2code.data_model import Tag, LinearizedTag
from sketch2code.helpers import norm_rgb_imgs, shrink_img, viz_grid
from sketch2code.render_engine import RemoteRenderEngine


class Action:
    def exec(self, state: 'Observation') -> Optional[LinearizedTag]:
        raise NotImplementedError()


class EnvCreator:
    def __init__(self, tags: List[Tag], sketches: List[np.ndarray], shrink_factor: float):
        self.render_engine: RemoteRenderEngine = RemoteRenderEngine.get_instance(tags[0].to_html(), sketches[0].shape[1],
                                                                                 sketches[0].shape[0])

        self.actions: List[Action] = [AddCloseTagAction()]
        for tag, classes in tags[0].supported_tags.items():
            self.actions.append(AddOpenTagAction(tag))
            for cls in classes:
                self.actions.append(AddClassAction(tag, cls))

        self.tags = tags
        self.shrink_factor = shrink_factor
        assert shrink_factor <= 1
        if shrink_factor != 1:
            self.sketches = [shrink_img(img, shrink_factor) for img in sketches]
        else:
            self.sketches = sketches

    def create(self):
        return [Env(self, tag, sketch) for tag, sketch in zip(self.tags, self.sketches)]

    # def render_imgs(self, tags: List[LinearizedTag]):
    #     imgs = self.render_engine.render_pages(tags)
    #     imgs = norm_rgb_imgs(imgs)
    #
    #     return imgs

    def render_img(self, tag: LinearizedTag):
        img = self.render_engine.render_page(tag)
        img = norm_rgb_imgs(img)
        if self.shrink_factor != 1:
            return shrink_img(img, self.shrink_factor)
        return img


class Observation:
    def __init__(self, goal_state: np.ndarray, goal_tag: Tag, curr_state: np.ndarray, curr_tag: LinearizedTag):
        self.goal_tag = goal_tag
        self.goal_state = goal_state
        self.curr_tag = curr_tag
        self.curr_state = curr_state


class Env:
    def __init__(self, env_creator: EnvCreator, goal_tag: Tag, sketch: np.ndarray):
        self.env_creator = env_creator
        self.goal_tag = Tag

        self.obs = Observation(sketch, goal_tag, np.zeros_like(sketch), LinearizedTag.default())
        self.matching_score = compute_matching_score(self.obs.goal_state, self.obs.curr_state)
        self.max_matching_score = np.prod(sketch.shape)

    def render(self):
        return viz_grid(np.stack([self.obs.goal_state, self.obs.curr_state], axis=0))

    def reset(self):
        self.obs = Observation(self.obs.goal_state, self.goal_tag, np.zeros_like(self.obs.goal_state), LinearizedTag.default())
        self.matching_score = compute_matching_score(self.obs.goal_state, self.obs.curr_state)
        self.max_matching_score = np.prod(self.obs.goal_state.shape)

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        new_tag = action.exec(self.obs)

        if new_tag is None:
            return self.obs, 0, False, {}

        new_sketch = self.env_creator.render_img(new_tag)
        new_obs = Observation(self.obs.goal_state, self.obs.goal_tag, new_sketch, new_tag)
        new_score = compute_matching_score(self.obs.goal_state, new_sketch)

        # compute the reward
        if isinstance(action, AddCloseTagAction):
            reward = 0
            done = new_tag.is_valid() and new_score == self.max_matching_score
        else:
            reward = (new_score - self.matching_score)
            done = isinstance(action, AddClassAction) and new_score == self.max_matching_score

        self.obs = new_obs
        self.matching_score = new_score
        
        return new_obs, reward, done, {}


def compute_matching_score(goal_state: np.ndarray, curr_state: np.ndarray) -> float:
    return (goal_state == curr_state).sum()


class AddOpenTagAction(Action):
    def __init__(self, tag_name: str):
        self.tag_name = tag_name

    def exec(self, state: Observation) -> Optional[LinearizedTag]:
        tag = state.curr_tag.clone()
        tag.add_open_tag(self.tag_name)
        return tag


class AddCloseTagAction(Action):
    def exec(self, state: Observation) -> Optional[LinearizedTag]:
        tag = state.curr_tag.clone()
        if not tag.add_close_tag():
            return None
        return tag


class AddClassAction(Action):
    def __init__(self, tag: str, cls: str):
        self.tag = tag
        self.cls = cls

    def exec(self, state: Observation) -> Optional[LinearizedTag]:
        tag = state.curr_tag.clone()
        if not tag.add_class(self.tag, self.cls):
            return None
        return tag
