#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import *

import numpy as np
import torch


def wrap_next_token_func(real_next_token_func):
    def exec(img, programs):
        program_lens = torch.tensor([len(x) for x in programs], device=img.device)
        programs = torch.tensor(programs, device=img.device)
        img = img.view(1, *img.shape).expand(programs.shape[0], *img.shape)
        return real_next_token_func(img, programs, program_lens)
    return exec


token2tag_re = re.compile("<div class=")
def token2tag(token: str):
    if 


def beam_search(img: np.ndarray, vocab, next_token_func, top_k, max_depth: int=100, device=None):
    """
    :param img: is already in C x W x H
    :param vocab:
    :param next_token_func:
    :param top_k:
    :param max_depth:
    :param device:
    :return:
    """
    programs = [([vocab['<div class="container-fluid">']], 1)]
    results = []

    for i in range(max_depth):
        gui = torch.tensor(img, device=device)

        next_tokens = next_token_func(gui, [x[0] for x in programs])
        next_token_probs, next_tokens = torch.sort(next_tokens, descending=True)
        
        next_token_probs = next_token_probs.view(len(programs), -1, next_tokens.shape[-1])
        next_tokens = next_tokens.view(len(programs), -1, next_tokens.shape[-1])
        
        next_programs = []
        for i in range(next_tokens.shape[0]):
            program, program_prob = programs[i]
            for j in range(top_k):
                next_token = next_tokens[i, -1, j]
                next_token_prob = next_token_probs[i, -1, j]

                new_program = program + [next_token.item()]
                new_program_prob = program_prob * next_token_prob.item()

                if next_token == vocab['<end>']:
                    results.append((new_program, new_program_prob))
                    continue

                next_programs.append((new_program, new_program_prob))

        next_programs.sort(key=lambda x: x[1], reverse=True)
        programs = next_programs[:top_k]
        if len(results) == top_k:
            break
    if len(results) == 0:
        results = programs[:top_k]
        
    return results

def html_css_syntax(program: List[int], ivocab: Dict[int, str]):
    for prev_token, token in zip(program[:-1], program[1:]):
        prev_token = ivocab[prev_token]
        token = ivocab[token]
        
        if 
    