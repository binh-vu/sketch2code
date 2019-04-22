#!/usr/bin/python
# -*- coding: utf-8 -*-


def conv2d_size_out(size, kernel_size, stride):
    """
    Number of Linear input connections depends on output of conv2d layers
    and therefore the input image size, so compute it.
    """
    return (size - (kernel_size - 1) - 1) // stride + 1


def pool2d_size_out(size, kernel_size, stride, padding=0, dilation=1):
    return (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
