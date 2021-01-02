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


def build_tokenizer_map_func(tokenizer, max_length=512):
    def tokenizer_map_func(rec):
        return tokenizer(rec['text_a'], rec['text_b'],
                         padding='max_length',
                         max_length=max_length,
                         truncation=True, )
    return tokenizer_map_func
