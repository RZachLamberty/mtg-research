#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: utils
Created: 2021-01-01

Description:

    shared utilities for mtg bert language modelling

Usage:

    >>> import utils

"""
import itertools


def build_tokenizer_map_func(tokenizer, max_length=512):
    def tokenizer_map_func(rec):
        return tokenizer(rec['text_a'], rec['text_b'],
                         padding='max_length',
                         max_length=max_length,
                         truncation=True,
                         return_special_tokens_mask=True)
    return tokenizer_map_func


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)
