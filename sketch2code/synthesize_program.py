#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import *

import numpy as np
from tqdm.autonotebook import tqdm
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
    OPEN_TAG = 0
    CLOSE_TAG = 1
    SPECIAL_TOKEN = 2

    invalid_parents = {
        "button": {"button"},
        "div": {"button"},
        "nav": {"button"},
        "program": {"div", "button", "nav", "a", "#text"},
        "h5": {"button"},
        "p": {"button"}
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
        tags = self.tags.append((tag, self.OPEN_TAG, classes))
        probs = self.tprobs.append(prob)

        return HTMLProgram(tags, opening_tags, self.prob * prob, probs)

    def add_close_tag(self, tag: str, prob: float):
        if len(self.opening_tags) == 0 or self.tags[self.opening_tags[-1]][0] != tag:
            return None

        opening_tags = self.opening_tags.delete(len(self.opening_tags) - 1)
        tags = self.tags.append((tag, self.CLOSE_TAG))
        probs = self.tprobs.append(prob)

        return HTMLProgram(tags, opening_tags, self.prob * prob, probs)

    def add_special_token(self, token, prob: float):
        tags = self.tags.append((token, self.SPECIAL_TOKEN, ()))
        probs = self.tprobs.append(prob)
        return HTMLProgram(tags, self.opening_tags, self.prob * prob, probs)

    def to_int_tokens(self, vocab: Dict[str, int]):
        int_tokens = []
        for tag in self.tags:
            if tag[1] == self.OPEN_TAG:  # is opening
                if len(tag[2]) == 0:
                    int_tokens.append(vocab[f'<{tag[0]}>'])
                else:
                    int_tokens.append(vocab[f'<{tag[0]} class="{" ".join(tag[2])}">'])
            elif tag[1] == self.CLOSE_TAG:
                int_tokens.append(vocab[f'</{tag[0]}>'])
            else:
                assert tag[1] == self.SPECIAL_TOKEN
                int_tokens.append(vocab[tag[0]])

        return int_tokens

    @staticmethod
    def from_int_tokens(int_tokens, ivocab: Dict[str, int]):
        program = HTMLProgram.default()
        for token in int_tokens:
            token = ivocab[token]
            tag, tag_type, classes = HTMLProgram.token2tag(token)
            if tag_type == self.OPEN_TAG:
                program = program.add_tag(tag, classes, 1.0)
            elif tag_type == self.CLOSE_TAG:
                program = program.add_close_tag(tag, 1.0)
            else:
                assert tag_type == self.SPECIAL_TOKEN
                program = program.add_special_token(tag, 1.0)
        return program

    @staticmethod
    def token2tag(token: str):
        if token.startswith("</"):
            return token[2:-1], HTMLProgram.CLOSE_TAG, None
        else:
            if token == "#text":
                return token, HTMLProgram.SPECIAL_TOKEN, None

            match = LinearizedTag.tag_reg.match(token)
            tag = match.group(1)
            if match.group(2) is None:
                classes = ()
            else:
                classes = match.group(2).split(" ")

            return tag, HTMLProgram.OPEN_TAG, classes

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
            if tag[0] == 'program':
                continue

            if tag[1] == self.OPEN_TAG:
                ltag.add_tag_and_class(tag[0], tag[2])
            elif tag[1] == self.CLOSE_TAG:
                ltag.add_close_tag()
            else:
                assert tag[1] == self.SPECIAL_TOKEN
                ltag.add_text(tag[0])
        return ltag

    def render_img(self, render_engine):
        return render_engine.render_page(self.to_linearized_tag())


def preprocess_image(preprocessed_img: np.ndarray, device=None):
    img = shrink_img(preprocessed_img, 0.5).transpose((2, 0, 1))
    return torch.tensor(img, device=device)


def compare_programs(programs: List[HTMLProgram], gui: np.ndarray,
                     render_engine: RemoteRenderEngine):
    imgs = render_engine.render_pages([p.to_linearized_tag() for p in programs])
    return (np.asarray(imgs) == gui).reshape((len(programs), -1)).sum(axis=1) / np.prod(gui.shape)


def synthesize(
        target_img: torch.tensor,
        ivocab,
        vocab,
        next_token_func: Callable[[torch.tensor, List[List[int]]], List[List[Tuple[int, float]]]],
        should_stop: Callable[[List[HTMLProgram]], bool],
        search_guidance: Callable[[List[HTMLProgram]], List[HTMLProgram]],
        branch_factor: int,
        beam_width: int,
        max_depth: int = 100,
        report_tqdm: bool = False):
    """
    :param target_img: a preprocessed image of the gui, which is already in C x W x H
    :param ivocab: inverse of vocabulary
    :param vocab: vocabulary
    :param next_token_func: return list of next tokens and its prob
    :param should_stop: telling if we have found the desired program and we should stop searching
    :param search_guidance: modifying the search states using information got from executing the program
    :param branch_factor:
    :param beam_width:
    :param max_depth:
    :param report_tqdm:
    :return:
    """
    programs: List[HTMLProgram] = [HTMLProgram.default().add_tag("program", (), 1.0)]
    results = []

    for _ in (tqdm(range(max_depth)) if report_tqdm else range(max_depth)):
        ntss: List[List[Tuple[int, float]]] = next_token_func(
            target_img, [p.to_int_tokens(vocab) for p in programs], top_k=branch_factor)

        next_programs = []
        assert len(ntss) == len(programs)
        for nts, program in zip(ntss, programs):
            for j, (nt, nt_prob) in enumerate(nts):
                nt = ivocab[nt]
                if j > 0 and nts[j - 1][1] / nt_prob >= 5:
                    # the gap is too huge
#                     print('>>>', nt, [(ivocab[x], y) for x, y in nts])
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

                tag, tag_type, classes = HTMLProgram.token2tag(nt)
                if tag_type == HTMLProgram.OPEN_TAG:
                    next_program = program.add_tag(tag, classes, nt_prob)
                elif tag_type == HTMLProgram.CLOSE_TAG:
                    next_program = program.add_close_tag(tag, nt_prob)
                else:
                    assert tag_type == HTMLProgram.SPECIAL_TOKEN
                    next_program = program.add_special_token(tag, nt_prob)
                
#                 if next_program is None:
#                     print("**", program.to_linearized_tag().str_tokens, nt, nt_prob)
#                     print("     --", nts)
#                     print("     --", next_token_func(target_img, [program.to_int_tokens(vocab)], top_k=branch_factor))
                if next_program is None:
                    # invalid program, so we have to ignore this token
                    continue

                next_programs.append(next_program)

        next_programs.sort(key=lambda x: x.prob, reverse=True)
        # execute the program to obtain more information and modify the next programs we should exploit to
        next_programs = search_guidance(next_programs)
        programs = next_programs[:beam_width]

        if should_stop(results) or len(results) == beam_width or len(programs) == 0:
            # no more program
            break

    if len(results) == 0:
        results = programs[:beam_width]
    return results
