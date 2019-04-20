import math
import random
from collections import namedtuple
from itertools import chain
from typing import List, Callable

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_value_
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm.autonotebook import tqdm

from sketch2code.config import ROOT_DIR
from sketch2code.helpers import inc_folder_no
from sketch2code.methods.lstm import prepare_batch_sentences, padded_aware_nllloss
from sketch2code.render_engine import RemoteRenderEngine
from sketch2code.synthesize_program import HTMLProgram

"""
    Contains utils for our supervised baseline: CNN (Encoder) -> RNN (Decoder) architecture
"""

Example = namedtuple('Example', ['img_idx', 'context_tokens', 'next_tokens'])


def make_toy_vocab_v1():
    """
    The first version of a vocabulary for the toy dataset.
    We merge class with tag to reduce the size of the programs
    """
    _tokens = []
    for c in ["row", "col-12", "col-6", "col-4", "col-3", "container-fluid", "grey-background"]:
        _tokens.append(f'<div class="{c}">')
    _tokens.append(f"</div>")
    for c in ["btn-danger", "btn-warning", "btn-success"]:
        _tokens.append(f'<button class="btn {c}">')
    _tokens.append(f"</button>")

    vocab = {'<pad>': 0, '<program>': 1, '</program>': 2}
    for i, token in enumerate(_tokens, start=len(vocab)):
        vocab[token] = i

    assert len(vocab) == len(_tokens) + 3
    ivocab = {v: k for k, v in vocab.items()}
    return vocab, ivocab


def make_pix2code_vocab_v1():
    """
    The first version of a vocabulary for the pix2code dataset, which follow the same principle in the toy v1 vocabulary
    """
    _tokens = []
    for c in ["row", "col-12", "col-6", "col-3", "container-fluid", "grey-background"]:
        _tokens.append(f'<div class="{c}">')
    
    for c in ["btn", "btn-danger", "btn-warning", "btn-success", "btn-primary", "btn-secondary"]:
        _tokens.append(f'<button class="btn {c}">')
    
    for w in ["</div>", "</button>", "<nav>", "</nav>", "<h5>", "</h5>", "<p>", "</p>", "#text"]:
        _tokens.append(w)
    
    vocab = {'<pad>': 0, '<program>': 1, '</program>': 2}
    for i, token in enumerate(_tokens, start=len(vocab)):
        vocab[token] = i

    assert len(vocab) == len(_tokens) + 3
    ivocab = {v: k for k, v in vocab.items()}
    return vocab, ivocab


def make_dataset_v1(example_indices, tags, dsl_vocab) -> List[Example]:
    """Create datasets using the vocabulary of the same version"""
    examples = []
    for i in example_indices:
        img_idx = i
        tag = tags[i]
        program = [
            dsl_vocab[x] for x in chain(['<program>'],
                                        tag.linearize().str_tokens, ['</program>'])
        ]
        examples.append(Example(img_idx, program[:-1], program[1:]))

    print("#examples", len(examples))
    return examples


def get_toy_dataset_v1(tags, vocab):
    indices = list(range(len(tags)))

    train_examples = make_dataset_v1(indices[:1250], tags, vocab)
    valid_examples = make_dataset_v1(indices[1250:1500], tags, vocab)
    test_examples = make_dataset_v1(indices[1500:], tags, vocab)

    return train_examples, valid_examples, test_examples


def iter_batch(batch_size: int,
               images: torch.tensor,
               examples: List[Example],
               shuffle: bool = False,
               device=None):
    """Loop through examples and generate batches"""

    index = list(range(len(examples)))
    if shuffle:
        random.shuffle(index)

    for i in range(0, len(examples), batch_size):
        batch_examples = [examples[j] for j in index[i:i + batch_size]]

        bx, bnx, bxlen, sorted_idx = prepare_batch_sentences(
            [e.context_tokens for e in batch_examples], [e.next_tokens for e in batch_examples],
            device=device)
        bimgs = images[[batch_examples[i].img_idx for i in sorted_idx]]
        yield (bimgs, bx, bnx, bxlen, sorted_idx)


class AverageMeter(object):
    """Keeps track of most recent, average, sum, and count of a metric."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seq_accuracy(bnx_pred, bnx, top_ks: List[int]):
    """
    Computes top-k accuracy, from predicted and true labels (packed padded sequence).
    :param bnx_pred: scores from the model: N_tokens x V
    :param bnx: true labels: N_tokens x V
    :param top_ks: different k (sorted descending)
    :return:
    """
    batch_size = bnx_pred.shape[0]
    bnx = bnx.view(-1, 1)
    _, ind = bnx_pred.topk(top_ks[-1], 1, True, True)
    accuracies = []
    for k in top_ks:
        acc = ind[:, :k].eq(bnx.expand((batch_size, k)))
        accuracies.append(float(acc.sum()) / batch_size)
    return accuracies


def evaluate(model, loss_func, images, examples, device=None, batch_size=500):
    loss = AverageMeter()
    top_1_acc = AverageMeter()
    top_3_acc = AverageMeter()
    top_5_acc = AverageMeter()

    model.eval()
    with torch.no_grad():
        for bimgs, bx, bnx, bxlen, sorted_idx in iter_batch(
                batch_size, images, examples, device=device):
            bnx_pred = model(bimgs, bx, bxlen)  # N x T x V
            bnx_pred, _ = pack_padded_sequence(bnx_pred, bxlen, batch_first=True)
            bnx, _ = pack_padded_sequence(bnx, bxlen, batch_first=True)

            n_tokens = bnx.shape[0]
            top_k_acc = seq_accuracy(bnx_pred, bnx, [1, 3, 5])

            loss.update(float(loss_func(bnx_pred, bnx)), n_tokens)
            top_1_acc.update(top_k_acc[0], n_tokens)
            top_3_acc.update(top_k_acc[1], n_tokens)
            top_5_acc.update(top_k_acc[2], n_tokens)
    model.train()
    return {'loss': loss, "top_1_acc": top_1_acc, "top_3_acc": top_3_acc, "top_5_acc": top_5_acc}


def train(model: nn.Module,
          loss_func,
          scheduler,
          optimizer,
          images,
          datasets,
          n_epoches: int,
          batch_size: int,
          clip_grad_val: float,
          eval_batch_size: int = 500,
          eval_valid_freq: int = 1,
          eval_test_freq: int = 3,
          device=None,
          exp_dir: str = "exp"):
    log_dir = inc_folder_no(ROOT_DIR / "runs" / exp_dir / "run_")
    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0
    model.train()

    valid_res = {'loss': AverageMeter(), 'top_1_acc': AverageMeter(), 'top_3_acc': AverageMeter(), 'top_5_acc': AverageMeter()}
    test_res = {'loss': AverageMeter(), 'top_1_acc': AverageMeter(), 'top_3_acc': AverageMeter(), 'top_5_acc': AverageMeter()}
    best_performance = 0.0

    train_examples, valid_examples, test_examples = datasets

    try:
        with tqdm(
                range(n_epoches), desc='epoch---') as epoches, tqdm(
                    total=math.ceil(len(train_examples) / batch_size) * n_epoches,
                    desc='training-') as pbar:
            for i in epoches:
                scheduler.step()
                batch_loss = AverageMeter()
                batch_top_1_acc = AverageMeter()
                batch_top_3_acc = AverageMeter()

                for bimgs, bx, bnx, bxlen, sorted_idx in iter_batch(
                        batch_size, images, train_examples, shuffle=True, device=device):
                    pbar.update()
                    global_step += 1

                    model.zero_grad()

                    bnx_pred = model(bimgs, bx, bxlen)
                    bnx_pred, _ = pack_padded_sequence(bnx_pred, bxlen, batch_first=True)
                    bnx, _ = pack_padded_sequence(bnx, bxlen, batch_first=True)

                    loss = loss_func(bnx_pred, bnx)
                    loss.backward()
                    clip_grad_value_(model.parameters(), clip_grad_val)  # prevent explode
                    optimizer.step()

                    loss = float(loss)
                    top_k_acc = seq_accuracy(bnx_pred, bnx, [1, 3])

                    writer.add_scalar('train/loss', loss, global_step)
                    writer.add_scalar('train/top_1_acc', top_k_acc[0], global_step)
                    writer.add_scalar('train/top_3_acc', top_k_acc[1], global_step)

                    n_tokens = bnx.shape[0]
                    batch_loss.update(loss, n_tokens)
                    batch_top_1_acc.update(top_k_acc[0], n_tokens)
                    batch_top_3_acc.update(top_k_acc[1], n_tokens)

                    pbar.set_postfix(loss=f"{loss:.5f}", top_1_acc=f"{top_k_acc[0]:.5f}", top_3_acc=f"{top_k_acc[1]:.5f}")

                if (i + 1) % eval_valid_freq == 0:
                    valid_res = evaluate(
                        model, loss_func, images, valid_examples, device, batch_size=eval_batch_size)
                    for k, v in valid_res.items():
                        writer.add_scalar(f'valid/{k}', v.avg, global_step)

                    if valid_res['top_1_acc'].avg > best_performance:
                        best_performance = valid_res['top_1_acc'].avg
                        torch.save({
                            "epoch": i,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }, log_dir + f"/model.bin")

                if (i + 1) % eval_test_freq == 0:
                    test_res = evaluate(
                        model, loss_func, images, test_examples, device, batch_size=eval_batch_size)
                    for k, v in test_res.items():
                        writer.add_scalar(f'test/{k}', v.avg, global_step)

                epoches.set_postfix(
                    b_l=f'{batch_loss.avg:.5f}',
                    b_a1=f'{batch_top_1_acc.avg:.5f}',
                    b_a3=f'{batch_top_3_acc.avg:.5f}',
                    v_l=f'{valid_res["loss"].avg:.5f}',
                    v_a1=f'{valid_res["top_1_acc"].avg:.5f}',
                    v_a3=f'{valid_res["top_3_acc"].avg:.5f}',
                    v_a5=f'{valid_res["top_5_acc"].avg:.5f}',
                    t_l=f'{test_res["loss"].avg:.5f}',
                    t_a1=f'{test_res["top_1_acc"].avg:.5f}',
                    t_a3=f'{test_res["top_3_acc"].avg:.5f}',
                    t_a5=f'{test_res["top_5_acc"].avg:.5f}',
                )
    finally:
        writer.close()


if __name__ == '__main__':
    """Test if these utility functions are running correctly"""
    import numpy as np
    from sketch2code.datasets import load_dataset
    from sketch2code.helpers import *

    tags, oimages = load_dataset("toy")
    print("#examples", len(tags))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    render_engine = RemoteRenderEngine.get_instance(tags[0].to_html(), 480, 300)
    vocab, ivocab = make_toy_vocab_v1()
    
    def preprocess_img():
        global oimages
        return [shrink_img(img, 0.4, cv2.INTER_NEAREST).transpose((2, 0, 1)) for img in norm_rgb_imgs(oimages[:])]

    images = cache_object("toy.shrink.imgs", preprocess_img)
    images = torch.tensor(images, device=device)
    train_examples, valid_examples, test_examples = get_toy_dataset_v1(tags, vocab)
    
    # TODO: uncomment to test the data
#     def verify_dataset(oimages, examples, example2img: Callable[[Example], np.ndarray]):
#         for e in tqdm(examples):
#             eimg = example2img(e)
#             gimg = oimages[e.img_idx]

#             matches = ((eimg == gimg).sum() / np.prod(eimg.shape))
#             assert matches == 1.0

#     example2img = lambda e: HTMLProgram.from_int_tokens(e.context_tokens, ivocab).render_img(render_engine)
#     verify_dataset(oimages, train_examples, example2img)
#     verify_dataset(oimages, valid_examples, example2img)
#     verify_dataset(oimages, test_examples, example2img)
    
    # TODO: uncomment to double check if iter_batch function is correct
    for examples in [train_examples, valid_examples, test_examples]:
        for bimgs, bx, bnx, bxlen, sorted_idx in tqdm(iter_batch(200, images, examples, True, device)):
            for gimg, x in tqdm(zip(bimgs, bx), total=200):
                p = HTMLProgram.from_int_tokens([w for w in x.tolist() if w > 0], ivocab)

                gimg = gimg.permute((1, 2, 0)).clone().cpu().numpy()
                eimg = shrink_img(norm_rgb_imgs(p.render_img(render_engine)), 0.4, cv2.INTER_NEAREST)
                matches = ((eimg == gimg).sum() / np.prod(eimg.shape))
                assert matches == 1.0, matches