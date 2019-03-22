#!/usr/bin/python
# -*- coding: utf-8 -*-
import copy

import numpy as np
from typing import *

from sketch2code.data_model import Tag, LinearizedTag
from sketch2code.render_engine import RenderEngine


class EnvCreator:

    def __init__(self, TagClass: Type[Tag], tags: List[Tag], sketches: List[np.ndarray]):
        self.render_engine = RenderEngine.get_instance(tags[0].to_html(), sketches[0].shape[1], sketches[0].shape[0])
        self.TagClass = TagClass

        self.actions = []
        for tag, classes in TagClass.supported_tags.items():
            self.actions.append(create_node(tag))
            for cls in classes:
                self.actions.append(add_class_to_node(cls))
            for pivot in ["move_in", "move_out"]:
                self.actions.append(move_pivot(pivot))

        self.tags = tags
        self.sketches = sketches

    def create(self):
        return [Env(self, tag, sketch) for tag, sketch in zip(self.tags, self.sketches)]


class State:

    def __init__(self, goal_state: np.ndarray, goal_tag: Tag, curr_state: np.ndarray, curr_tag: LinearizedTag):
        self.goal_tag = goal_tag
        self.goal_state = goal_state
        self.curr_tag = curr_tag
        self.curr_state = curr_state


class Env:

    def __init__(self, env_creator: EnvCreator, goal_tag: Tag, sketch: np.ndarray):
        self.env_creator = env_creator
        self.goal_tag = Tag

        self.state = State(sketch, goal_tag, np.ones_like(sketch), env_creator.TagClass("html", [], []))

    def step(self, action: Callable[[State], Tag]):
        new_tag, new_pivot = action(self.state)
        new_sketch = self.env_creator.render_engine.render_page(new_tag)
        new_state = State(self.state.goal_state, self.state.goal_tag, new_sketch, new_tag, )

        # compute the reward here


def create_node(name: str):
    def action(state: State):
        tag = state.curr_tag.clone()
        tag.opening_tags.append(len(tag.tokens))
        tag.tokens.append(f"<{name}>")

        return tag

    return action


def add_class_to_node(cls: str):
    def action(state: State):
        state

    return action


def move_pivot(action: str):
    def action(state: State):
        pass

    return action