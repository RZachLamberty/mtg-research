#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: edhrec_dataset_chunk_proc
Created: 2021-01-02

Description:

    simple runner for a dirty multiproc hack for edhrec dataset creation

Usage:

    >>> import edhrec_dataset_chunk_proc

"""
import argparse
import os

import datasets
from transformers import BertTokenizerFast

from utils import build_tokenizer_map_func


# turn off progress bars
from datasets.utils.logging import set_verbosity_error
set_verbosity_error()


def add_labels(examples):
    return {'labels': examples['input_ids']}


def main(num_chunks, chunk, data_dir, num_max_pairs, out_dir, tokenizer_model_name, max_seq_length):
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_model_name)
    tokenizer_map_func = build_tokenizer_map_func(tokenizer, max_length=max_seq_length)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    (datasets.load_dataset('edhrec_dataset.py',
                           data_dir=data_dir,
                           num_max_pairs=num_max_pairs,
                           num_chunks=num_chunks,
                           chunk=chunk)
     .map(tokenizer_map_func, batched=False)
     .map(add_labels, batched=True)
     .save_to_disk(f"{out_dir}/{chunk:0>4}-{num_chunks:0>4}"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process one chunk of the edh dataset')

    # required
    parser.add_argument('-n', '--num_chunks', help='number of chunks', type=int, required=True)
    parser.add_argument('-c', '--chunk', help='chunk number', type=int, required=True)
    parser.add_argument('-d', '--data_dir', help='data directory containing parquet files',
                        required=True)
    parser.add_argument('-o', '--out_dir', help='directory to save datasets in', required=True)

    # optional
    parser.add_argument('-m', '--num_max_pairs', help='max number of pairs per card', type=int)
    parser.add_argument('-t', '--tokenizer_model_name', help='dir name of saved trained tokenizer',
                        default='mtg-language')
    parser.add_argument('-l', '--max_seq_length', help='maximum token sequence length', type=int,
                        default=200)

    args = parser.parse_args()

    assert 0 <= args.chunk < args.num_chunks

    main(num_chunks=args.num_chunks,
         chunk=args.chunk,
         data_dir=args.data_dir,
         num_max_pairs=args.num_max_pairs,
         out_dir=args.out_dir,
         tokenizer_model_name=args.tokenizer_model_name,
         max_seq_length=args.max_seq_length)
