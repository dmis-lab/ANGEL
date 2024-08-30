from typing import Dict, Callable, Iterable, List, Optional
# from transformers.generation_logits_process import LogitsProcessor
# import torch
# import math
try:
    import marisa_trie
except ModuleNotFoundError:
    pass

# class PrefixConstrainedLogitsProcessor(LogitsProcessor):

#     def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int):
#         self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
#         self._num_beams = num_beams

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         mask = torch.full_like(scores, -math.inf)
#         for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
#             for beam_id, sent in enumerate(beam_sent):
#                 mask[batch_id * self._num_beams + beam_id, self._prefix_allowed_tokens_fn(batch_id, sent)] = 0
#         return scores + mask




class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)


class DummyTrieMention(object):
    def __init__(self, return_values):
        self._return_values = return_values

    def get(self, indices=None):
        return self._return_values


class DummyTrieEntity(object):
    def __init__(self, return_values, codes):
        self._return_values = list(
            set(return_values).difference(
                set(
                    codes[e]
                    for e in (
                        "start_mention_token",
                        "end_mention_token",
                        "start_entity_token",
                    )
                )
            )
        )
        self._codes = codes

    def get(self, indices, depth=0):
        if len(indices) == 0 and depth == 0:
            return self._codes["end_mention_token"]
        elif len(indices) == 0 and depth == 1:
            return self._codes["start_entity_token"]
        elif len(indices) == 0:
            return self._return_values
        elif len(indices) == 1 and indices[0] == self._codes["end_entity_token"]:
            return self._codes["EOS"]
        else:
            return self.get(indices[1:], depth=depth + 1)
