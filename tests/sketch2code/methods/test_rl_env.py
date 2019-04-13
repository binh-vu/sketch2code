#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import *

from sketch2code.data_model import LinearizedTag, ToyTag
from sketch2code.methods.rl_env import EnvCreator, AddOpenTagAction, AddCloseTagAction, Observation, AddClassAction, \
    Action, compute_semantic_penalty, AddOpenTagAndClassAction


def action_exec_should_success(obs: Observation, action: Action):
    assert action.is_valid(obs)
    obs.curr_tag = action.exec(obs)
    assert obs.curr_tag is not None


def action_exec_should_fail(obs: Observation, action: Action):
    assert action.exec(obs) is None
    assert not action.is_valid(obs)


def test_add_open_tag():
    tag = LinearizedTag.default()

    btnOpen = AddOpenTagAction(0, "button")
    btnClose = AddCloseTagAction(1)

    obs = Observation(None, None, None, None, tag)

    action_exec_should_success(obs, btnOpen)
    action_exec_should_success(obs, btnClose)
    action_exec_should_fail(obs, btnClose)


def test_add_class_semantic_penalty():
    tag = LinearizedTag.default()
    obs = Observation(None, None, None, None, tag)

    # test add class
    btnOpen = AddOpenTagAction(0, "button")
    btnSuccess = AddClassAction(3, "button", "btn-success")
    btnDanger = AddClassAction(3, "button", "btn-danger")

    action_exec_should_success(obs, btnOpen)
    assert compute_semantic_penalty(obs, btnSuccess, -50) == 0
    action_exec_should_success(obs, btnSuccess)
    assert compute_semantic_penalty(obs, btnDanger, -50) == -50


def test_add_open_tag_semantic_penalty():
    tag = LinearizedTag.default()
    obs = Observation(None, None, None, None, tag)

    # test add class
    btnOpen = AddOpenTagAction(0, "button")
    divOpen = AddOpenTagAction(0, "div")

    action_exec_should_success(obs, btnOpen)
    assert compute_semantic_penalty(obs, divOpen, -50) == -50


def test_semantic_penalty():
    obs = Observation(None, None, None, None, None)
    obs.curr_tag = LinearizedTag.default()

    container = AddOpenTagAndClassAction(0, "div", ("container-fluid",))
    row = AddOpenTagAndClassAction(0, "div", ("row",))
    col = AddOpenTagAndClassAction(0, "div", ("col-4",))
    box = AddOpenTagAndClassAction(0, "div", ("grey-background",))

    assert compute_semantic_penalty(obs, container, -50) == 0
    action_exec_should_success(obs, container)
    assert compute_semantic_penalty(obs, row, -50) == 0
    action_exec_should_success(obs, row)
    assert compute_semantic_penalty(obs, col, -50) == 0
    action_exec_should_success(obs, col)
    assert compute_semantic_penalty(obs, container, -50) == -50
    assert compute_semantic_penalty(obs, box, -50) == 0
    action_exec_should_success(obs, box)
