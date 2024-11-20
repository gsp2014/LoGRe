from tqdm import tqdm
from collections import defaultdict
import numpy as np
from typing import DefaultDict, List, Tuple, Dict, Set

def create_adj_list(file_name: str) -> DefaultDict[str, List[Tuple[str, str]]]:
    out_map = defaultdict(list)
    fin = open(file_name)
    for line_ctr, line in tqdm(enumerate(fin)):
        line = line.strip()
        e1, e2, r = line.split("\t")
        out_map[e1].append((r, e2))
    fin.close()
    return out_map

def load_data(file_name: str) -> DefaultDict[Tuple[str, str], list]:
    out_map = defaultdict(list)
    fin = open(file_name)
    for line in tqdm(fin):
        line = line.strip()
        e1, e2, r = line.split("\t")
        out_map[(e1, r)].append(e2)
    fin.close()
    return out_map

def load_data_all_triples(train_file: str, dev_file: str, test_file: str) -> DefaultDict[Tuple[str, str], list]:
    """
    Returns a map of all triples in the knowledge graph. Use this map only for filtering in evaluation.
    :param train_file:
    :param dev_file:
    :param test_file:
    :return:
    """
    out_map = defaultdict(list)
    for file_name in [train_file, dev_file, test_file]:
        fin = open(file_name)
        for line in tqdm(fin):
            line = line.strip()
            e1, e2, r = line.split("\t")
            out_map[(e1, r)].append(e2)
        fin.close()
    return out_map

def create_vocab(kg_file: str) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    entity_vocab, rev_entity_vocab = {}, {}
    rel_vocab, rev_rel_vocab = {}, {}
    fin = open(kg_file)
    entity_ctr, rel_ctr = 0, 0
    for line in tqdm(fin):
        line = line.strip()
        e1, e2, r = line.split("\t")
        if e1 not in entity_vocab:
            entity_vocab[e1] = entity_ctr
            rev_entity_vocab[entity_ctr] = e1
            entity_ctr += 1
        if e2 not in entity_vocab:
            entity_vocab[e2] = entity_ctr
            rev_entity_vocab[entity_ctr] = e2
            entity_ctr += 1
        if r not in rel_vocab:
            rel_vocab[r] = rel_ctr
            rev_rel_vocab[rel_ctr] = r
            rel_ctr += 1
    fin.close()
    return entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab

def read_graph(file_name: str, entity_vocab: Dict[str, int], rel_vocab: Dict[str, int]) -> np.ndarray:
    adj_mat = np.zeros((len(entity_vocab), len(rel_vocab)))
    fin = open(file_name)
    for line in tqdm(fin):
        line = line.strip()
        e1, e2, r = line.split("\t")
        adj_mat[entity_vocab[e1], rel_vocab[r]] = 1
    fin.close()
    return adj_mat

def get_unique_entities(kg_file: str) -> Set[str]:
    unique_entities = set()
    fin = open(kg_file)
    for line in fin:
        e1, e2, r = line.strip().split()
        unique_entities.add(e1)
        unique_entities.add(e2)
    fin.close()
    return unique_entities

def get_inv_relation(r: str, dataset_name="nell") -> str:
    if r[-4:] == "_inv":
        return r[:-4]
    else:
        return r + "_inv"
