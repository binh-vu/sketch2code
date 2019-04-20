#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import *

import numpy as np
import torch
from pyrsistent import PVector, pvector

from sketch2code.data_model import LinearizedTag
from sketch2code.helpers import shrink_img
from sketch2code.render_engine import RemoteRenderEngine


def wrap_next_token_func(real_next_token_func):
    def exec(img, tokens):
        program_lens = torch.tensor([len(p) for p in tokens], device=img.device)
        tokens = torch.tensor(tokens, device=img.device)
        img = img.view(1, *img.shape).expand(tokens.shape[0], *img.shape)
        return real_next_token_func(img, tokens, program_lens)
    return exec


class HTMLProgram:
    """
        Represent the HTML program in a sequence of tags
        (tag_name, is_opening_tag, [classes])
    """

    invalid_parents = {
        "button": {"button"},
        "div": {"button"}
    }

    def __init__(self, tags: PVector, opening_tags: PVector, prob: float, tprobs: PVector):
        self.tags = tags
        self.opening_tags = opening_tags
        self.prob = prob
        self.tprobs = tprobs
        self.quality: Optional[float] = None

    @staticmethod
    def default():
        return HTMLProgram(pvector([]), pvector([]), 1.0, pvector([]))

    def add_tag(self, tag: str, classes: Tuple[str, ...], prob: float):
        if len(self.opening_tags) > 0:
            parent_tag = self.tags[self.opening_tags[-1]][0]
            if parent_tag in HTMLProgram.invalid_parents[tag]:
                return None

        opening_tags = self.opening_tags.append(len(self.tags))
        tags = self.tags.append((tag, True, classes))
        probs = self.tprobs.append(prob)

        return HTMLProgram(tags, opening_tags, self.prob * prob, probs)

    def add_close_tag(self, tag: str, prob: float):
        if len(self.opening_tags) == 0 or self.tags[self.opening_tags[-1]][0] != tag:
            return None

        opening_tags = self.opening_tags.delete(len(self.opening_tags) - 1)
        tags = self.tags.append((tag, False))
        probs = self.tprobs.append(prob)

        return HTMLProgram(tags, opening_tags, self.prob * prob, probs)

    def to_int_tokens(self, vocab: Dict[str, int]):
        int_tokens = []
        for tag in self.tags:
            if tag[1]: # is opening
                if len(tag[2]) == 0:
                    int_tokens.append(vocab[f'<{tag[0]}>'])
                else:
                    int_tokens.append(vocab[f'<{tag[0]} class="{" ".join(tag[2])}">'])
            else:
                int_tokens.append(vocab[f'</{tag[0]}>'])

        return int_tokens

    @staticmethod
    def from_int_tokens(int_tokens, ivocab: Dict[str, int]):
        program = HTMLProgram.default()
        for token in int_tokens:
            token = ivocab[token]
            tag, is_open, classes = HTMLProgram.token2tag(token)
            if is_open:
                program = program.add_tag(tag, classes, 1.0)
            else:
                program = program.add_close_tag(tag, 1.0)
        return program

    @staticmethod
    def token2tag(token: str):
        if token.startswith("</"):
            return token[2:-1], False, None
        else:
            match = LinearizedTag.tag_reg.match(token)
            tag = match.group(1)
            if match.group(2) is None:
                classes = ()
            else:
                classes = match.group(2).split(" ")

            return tag, True, classes

    def print_next_tags(self, img: np.ndarray, next_token_func, ivocab, vocab, device=None):
        nts = next_token_func(torch.tensor(img, device=device), [self.to_int_tokens(vocab)])
        nt_probs, nts = torch.sort(nts, descending=True)

        nt_probs = nt_probs.view(1, -1, nts.shape[-1])[0, -1]
        nts = nts.view(1, -1, nts.shape[-1])[0, -1]

        print(f"token_idx | {'next_token_str': <50}| prob")
        for nt, nt_prob in zip(nts, nt_probs):
            print(f"{nt.item(): 9} | {ivocab[nt.item()]: <50}| {nt_prob.exp().item():.5f}")

    def to_linearized_tag(self):
        ltag = LinearizedTag.default()
        for tag in self.tags:
            if tag[0] == 'begin' or tag[0] == 'end':
                continue

            if tag[1]:
                ltag.add_tag_and_class(tag[0], tag[2])
            else:
                ltag.add_close_tag()
        return ltag

    def render_img(self, render_engine):
        return render_engine.render_page(self.to_linearized_tag())


def preprocess_image(preprocessed_img: np.ndarray, device=None):
    img = shrink_img(preprocessed_img, 0.5).transpose((2, 0, 1))
    return torch.tensor(img, device=device)


def compare_programs(programs: List[HTMLProgram], gui: np.ndarray, render_engine: RemoteRenderEngine):
    imgs = render_engine.render_pages([p.to_linearized_tag() for p in programs])
    return (np.asarray(imgs) == gui).reshape((len(programs), -1)).sum(axis=1) / np.prod(gui.shape)


def synthesize(gui: torch.tensor, ogui, render_engine, ivocab, vocab, next_token_func, top_k, beam_width: int, min_quality: float=0.9, max_unexamined_program: int=3, max_depth: int=100):
    """
    :param gui: a preprocessed image of the gui, which is already in C x W x H
    :param vocab:
    :param next_token_func:
    :param top_k:
    :param max_depth:
    :param device:
    :return:
    """
    programs: List[HTMLProgram] = [
        HTMLProgram.default().add_tag("program", (), 1.0)
    ]
    results = []

    for d in range(max_depth):
#         print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", d)
        # for p in programs[:5]:
        #     print('>>', f"{p.prob:.5f}", f"{p.quality if p.quality is not None else float('nan'):.5f}", [ivocab[w] for w in p.to_int_tokens(vocab)])

        nts = next_token_func(gui, [x.to_int_tokens(vocab) for x in programs])
        nt_probs, nts = torch.sort(nts, descending=True)

        nt_probs = nt_probs.view(len(programs), -1, nts.shape[-1])
        nts = nts.view(len(programs), -1, nts.shape[-1])

        next_programs = []
        for i in range(nts.shape[0]):
            program = programs[i]
            for j in range(top_k):
                nt = ivocab[int(nts[i, -1, j])]
                nt_prob = float(nt_probs[i, -1, j].exp())

                if j > 0 and nt_probs[i, -1, j - 1].exp() - nt_prob > 0.7:
                    # the gap is too huge
                    break

                if nt == "<program>":
                    # only happen at the beginning
                    continue
                elif nt == "</program>":
                    next_program = program.add_close_tag("program", nt_prob)
                    if next_program is not None:
                        results.append(next_program)
                    continue
                elif nt == "<pad>":
                    continue

                tag, is_opening, classes = HTMLProgram.token2tag(nt)
                if is_opening:
                    next_program = program.add_tag(tag, classes, nt_prob)
                else:
                    next_program = program.add_close_tag(tag, nt_prob)

                if next_program is None:
                    # invalid program, so we have to ignore this token
                    continue

                next_programs.append(next_program)

        # for p, q in zip(next_programs, compare_programs(next_programs, ogui, render_engine)):
        #     p.quality = q
        next_programs.sort(key=lambda x: x.prob, reverse=True)
        programs = next_programs[:beam_width]

        # we filter out the results list to eliminate not good program, we filter the list of undesired images
        unexamined_programs = [p for p in results if p.quality is None]
        if len(unexamined_programs) >= max_unexamined_program:
            qualities = compare_programs(unexamined_programs, ogui, render_engine)
            for i, q in enumerate(qualities):
                unexamined_programs[i].quality = q
            results = [p for p in results if p.quality >= min_quality]

        if len(results) == beam_width:
            break

        if len(programs) == 0:
            break

    if len(results) == 0:
        results = programs[:beam_width]

    unexamined_programs = [p for p in results if p.quality is None]
    if len(unexamined_programs) > 0:
        qualities = compare_programs(unexamined_programs, ogui, render_engine)
        for i, q in enumerate(qualities):
            unexamined_programs[i].quality = q

    # results = [p for p in results if p.quality >= min_quality]
    results.sort(key=lambda p: p.quality, reverse=True)
    return results
