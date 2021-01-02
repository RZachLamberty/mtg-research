import os
from dataclasses import dataclass

import datasets
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# turn off progress bars
from datasets.utils.logging import set_verbosity_error
set_verbosity_error()

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """
@InProceedings{huggingface:dataset,
title = mtg-edhrec,
authors={r.zach.lamberty@gmail.com
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """
MLM for MTG, plus card pair suggestions based off of edhrec crowdsourced data. 
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://github.com/RZachLamberty/mtg-research"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

SPLITS = ['train', 'test', 'validation']


@dataclass
class EdhrecConfig(datasets.BuilderConfig):
    num_max_pairs: int = None
    num_chunks: int = None
    chunk: int = None


class EdhrecDataset(datasets.GeneratorBasedBuilder):
    """generate pairs of cards with positive labels being edhrec-suggested card pairs"""

    BUILDER_CONFIG_CLASS = EdhrecConfig
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                'name_a': datasets.Value(dtype='string'),
                'text_a': datasets.Value(dtype='string'),
                'name_b': datasets.Value(dtype='string'),
                'text_b': datasets.Value(dtype='string'),
                'next_sentence_label': datasets.Value(dtype='int64'),
                'rec_set_type': datasets.Value(dtype='string'),
                'id': datasets.Value(dtype='string'), }),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION, )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'split': 'train'}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={'split': 'test'}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                    gen_kwargs={'split': 'validation'}),
        ]

    def _generate_examples(self, split):
        """actually yield examples from split"""
        #print(f"num_chunks = {self.config.num_chunks}")
        #print(f"chunk = {self.config.chunk}")
        assert (self.config.num_chunks is None) == (
                    self.config.chunk is None), "provide both num_chunks and chunk or neither"

        these_cards = pd.read_parquet(os.path.join(self.config.data_dir, f"cards.{split}.parquet"))
        these_edhrec = pd.read_parquet(os.path.join(self.config.data_dir,
                                                    f"edhrec.{split}.parquet"))
        non_edhrec_card_names = set(these_cards.index).difference(these_edhrec.name.unique())

        if self.config.num_chunks is None:
            a_recs = these_cards
        else:
            chunk_size = these_cards.shape[0] / self.config.num_chunks
            i0 = int(self.config.chunk * chunk_size)
            i1 = int((self.config.chunk + 1) * chunk_size)
            a_recs = these_cards.iloc[i0: i1]

        for name_a, card_row in tqdm(a_recs.iterrows(), total=a_recs.shape[0]):
            text_a = card_row.mytext
            sfid_a = card_row.scryfallId
            commanders = these_edhrec[these_edhrec.name == name_a].commander.unique()

            edhrec_peers = (these_edhrec
                            [these_edhrec.commander.isin(commanders)
                             & (these_edhrec.name != name_a)]
                            .name.unique())
            edhrec_non_peers = (these_edhrec
                                [(~these_edhrec.commander.isin(commanders))
                                 & (these_edhrec.name != name_a)]
                                .name.unique())
            non_edhrec_non_peers = non_edhrec_card_names

            if self.config.num_max_pairs is not None:
                edhrec_peers = np.random.choice(
                    edhrec_peers,
                    size=min(self.config.num_max_pairs, edhrec_peers.shape[0]),
                    replace=False)
                edhrec_non_peers = np.random.choice(
                    edhrec_non_peers,
                    size=min(self.config.num_max_pairs, edhrec_non_peers.shape[0]),
                    replace=False)
                non_edhrec_non_peers = np.random.choice(
                    list(non_edhrec_non_peers),
                    size=min(self.config.num_max_pairs, len(non_edhrec_non_peers)),
                    replace=False)

            rec_iters = [(edhrec_peers, 1, 'edh-peer'),
                         (edhrec_non_peers, 0, 'edh-non_peer'),
                         (non_edhrec_non_peers, 0, 'non_edh-non_peer')]

            for name_b_rec_set, next_sentence_label, rec_set_type in rec_iters:
                for name_b in name_b_rec_set:
                    try:
                        row_b = these_cards.loc[name_b]
                        text_b = row_b.text
                        sfid_b = row_b.scryfallId
                    except KeyError:
                        continue

                    id_ = f'{sfid_a}_{sfid_b}'
                    yield (id_,
                           {'name_a': name_a,
                            'text_a': text_a,
                            'name_b': name_b,
                            'text_b': text_b,
                            'next_sentence_label': next_sentence_label,
                            'rec_set_type': rec_set_type,
                            'id': id_, })
