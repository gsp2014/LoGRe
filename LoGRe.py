import argparse
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
import pickle
import torch
import uuid
from data_utils import get_unique_entities, create_adj_list, create_vocab, load_data, load_data_all_triples, read_graph, get_inv_relation
from get_paths import get_paths
from typing import *
from log import Logger
import json
import sys
import wandb

class LoGRe(object):
    def __init__(self, args, train_map, eval_map, entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, eval_vocab, eval_rev_vocab, all_paths):
        self.args = args
        self.eval_map = eval_map
        self.train_map = train_map
        self.all_zero_ctr = []
        self.all_num_ret_nn = []
        self.entity_vocab, self.rev_entity_vocab, self.rel_vocab, self.rev_rel_vocab = entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab
        self.eval_vocab, self.eval_rev_vocab = eval_vocab, eval_rev_vocab
        self.all_paths = all_paths
        self.num_non_executable_programs = []
        self.in_candidate = 0
        self.cached_neighbors = {}
        self.cached_programs, self.cached_siment = {}, {}
        self.cached_num_ret_nn, self.cached_zero_ctr = {}, {}
        self.cached_rank_results = {}

    # Cluster entities according to their types
    def cluster_entities_type(self, type_file):  #dict{e: type}, dict{type: set(e)}
        cluster_assignments_type, type2ents = {}, {}
        with open(type_file, "r") as fin:
            for _, line in enumerate(fin):
                e, t = line.strip().split("\t")
                cluster_assignments_type[e] = t
                if t not in type2ents:
                    type2ents[t] = set()
                type2ents[t].add(e)
        return cluster_assignments_type, type2ents
    
    def set_ansim(self, ansim):
        self.ansim = ansim
    
    # Get entities having the same r.
    def get_neighbors(self, r: str):
        # Update: add cache to save time
        if self.cached_neighbors.get(r) is None:
            nearest_entities = self.entity_vocab.keys()
            temp = []
            for nn in nearest_entities:
                if len(self.train_map[nn, r]) > 0:
                    temp.append(nn)
            self.cached_neighbors[r] = temp
        return self.cached_neighbors[r]

    # Given an entity and answer, get all paths ending at the answer node within the subgraph surrounding the entity.
    @staticmethod
    def get_programs(e: str, ans: str, all_paths_around_e: List[List[str]]):
        all_programs = []
        for path in all_paths_around_e:
            for l, (r, e_dash) in enumerate(path):
                if e_dash == ans:
                    # get the path till this point
                    all_programs.append([x for (x, _) in path[:l + 1]])  # we only need to keep the relations
        return all_programs

    # Given a relation r, run get_programs for entities having r and record the corresponding terminal entities of the paths for the calculation of answer similarity.
    def get_programs_from_nearest_neighbors(self, r: str, nn_func: Callable):
        # Update: add cache to save time
        if r in self.cached_programs:
            self.all_num_ret_nn.append(self.cached_num_ret_nn[r])
            self.all_zero_ctr.append(self.cached_zero_ctr[r])
            return self.cached_programs[r]
        all_programs = set()
        self.cached_siment[r] = {}
        nearest_entities = nn_func(r)
        if nearest_entities is None:
            self.all_num_ret_nn.append(0)
            self.cached_num_ret_nn[r] = 0
            return None
        self.all_num_ret_nn.append(len(nearest_entities))
        self.cached_num_ret_nn[r] = len(nearest_entities)
        zero_ctr = 0
        for e in nearest_entities:
            if len(self.train_map[(e, r)]) > 0:
                paths_e = self.all_paths[e]  # get the collected paths around e
                nn_answers = self.train_map[(e, r)]
                for nn_ans in nn_answers:
                    ps = self.get_programs(e, nn_ans, paths_e)
                    for p in ps:
                        # filter the program if it is equal to the query relation
                        if len(p) == 1 and p[0] == r:
                            continue
                        all_programs.add(tuple(p))
                        p_str = '-'.join(p)
                        if p_str not in self.cached_siment[r]:
                            self.cached_siment[r][p_str] = set()
                        self.cached_siment[r][p_str].add(nn_ans)
            elif len(self.train_map[(e, r)]) == 0:
                zero_ctr += 1
        all_programs = list(all_programs)
        self.all_zero_ctr.append(zero_ctr)
        self.cached_zero_ctr[r] = zero_ctr
        self.cached_programs[r] = all_programs
        return all_programs

    # Rank paths.
    def rank_programs(self, list_programs: List[List[str]], r: str):
        # Update: add cache to save time
        if r in self.cached_rank_results:
            return self.cached_rank_results[r]
        unique_programs = list_programs
        # now get the score of each path
        path_and_scores = []
        for p in unique_programs:
            try:
                path_and_scores.append((p, self.args.precision_map[r][p] * self.args.hop_factors[len(p)]))
            except KeyError:
                continue

        # sort paths by their scores
        sorted_programs = [k for k, v in sorted(path_and_scores, key=lambda item: -item[1])]
        self.cached_rank_results[r] = sorted_programs
        return sorted_programs

    # Starting from a given entity, execute the path by doing depth-first search. If there are multiple entities with the same relation, we select up to max_branch entities.
    def execute_one_program(self, e: str, path: List[str], depth: int, max_branch: int):
        if depth == len(path):
            # reached end, return node
            return [e]
        next_entities = self.train_map[(e, path[depth])]
        if len(next_entities) == 0:
            # edge not present
            return []
        if len(next_entities) > max_branch:
            # select max_branch random entities
            next_entities = np.random.choice(next_entities, max_branch, replace=False).tolist()
        answers = []
        for e_next in next_entities:
            answers += self.execute_one_program(e_next, path, depth + 1, max_branch)
        return answers

    # Given an entity, relation, and path list, run execute_one_program for each path.
    def execute_programs(self, e: str, r: str, path_list: List[List[str]], max_branch: int):
        all_answers = []
        not_executed_paths = []
        execution_fail_counter = 0
        executed_path_counter = 0
        for i in range(len(path_list)):
            path = path_list[i]
            if executed_path_counter == self.args.max_num_programs:
                break
            ans = self.execute_one_program(e, path, depth=0, max_branch=max_branch)
            temp = []
            try:
                path_score = self.args.precision_map[r][path] * self.args.hop_factors[len(path)]
            except KeyError:
                # either the path or relation is missing
                path_score = 0
            for a in ans:
                path = tuple(path)
                temp.append((a, path_score, path))
            ans = temp
            if ans == []:
                not_executed_paths.append(path)
                execution_fail_counter += 1
            else:
                executed_path_counter += 1
            all_answers += ans
        self.num_non_executable_programs.append(execution_fail_counter)
        return all_answers, not_executed_paths

    # Aggregate the answers, assigning each candidate answer a score by summing the scores of the paths reaching it. Record the corresponding terminal entities of the paths for the calculation of answer similarity.
    def aggregate_answers(self, r, list_answers: List[Tuple[str, float, List[str]]]):
        count_map = {}
        ent_map = {}
        for i in range(len(list_answers)):
            (e, e_score, path) = list_answers[i]
            if e not in count_map:
                count_map[e] = {}
                ent_map[e] = set()
            if path not in count_map[e]:
                count_map[e][path] = e_score  # just count once for a path type.
            ent_map[e] = ent_map[e] | self.cached_siment[r]['-'.join(path)]
        score_map = defaultdict(int)
        for e, path_scores_map in count_map.items():
            sum_path_score = 0
            for _, p_score in path_scores_map.items():
                sum_path_score += p_score
            score_map[e] = sum_path_score
        return score_map, ent_map
    
    @staticmethod
    def get_rank_in_list(e, predicted_answers):
        for i, e_to_check in enumerate(predicted_answers):
            if e == e_to_check[0]:
                return i + 1
        return -1

    # Consider answer similarity and get the metric counts.
    def get_hits(self, answers, gold_answers, all_siment, query):
        hits_1 = 0.0
        hits_3 = 0.0
        hits_5 = 0.0
        hits_10 = 0.0
        rr = 0.0
        mr = 0.0
        (e1, r) = query
        all_gold_answers = self.args.all_kg_map[(e1, r)]
        for gold_answer in gold_answers:
            # remove all other gold answers from prediction
            filtered_answers = {}
            for pred in answers:
                if pred in all_gold_answers and pred != gold_answer:
                    continue
                else:
                    ss = 0
                    for ent in all_siment[pred]:
                        tmp = self.ansim[self.entity_vocab[pred]][self.entity_vocab[ent]]
                        ss = max(ss, tmp)
                    filtered_answers[pred] = answers[pred] * ss
            sorted_filtered_answers = sorted(filtered_answers.items(), key=lambda kv: -kv[1])
            rank = LoGRe.get_rank_in_list(gold_answer, sorted_filtered_answers)
            if rank > 0:
                if rank <= 10:
                    hits_10 += 1
                    if rank <= 5:
                        hits_5 += 1
                        if rank <= 3:
                            hits_3 += 1
                            if rank <= 1:
                                hits_1 += 1
                rr += 1.0 / rank
                mr += rank
                self.in_candidate = self.in_candidate + 1
        return hits_10, hits_5, hits_3, hits_1, rr, mr

    def path_reasoning(self):
        num_programs = []
        num_answers = []
        non_zero_ctr = 0
        hits_10, hits_5, hits_3, hits_1, mrr, mr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        per_relation_scores = {}  # map of performance per relation
        per_relation_query_count = {}
        total_examples = 0
        learnt_programs = defaultdict(lambda: defaultdict(int))  # for each query relation, a map of programs to count
        for _, ((e1, r), e2_list) in enumerate(tqdm(self.eval_map.items())):
            # if e2_list is in train list then remove them
            # Normally, this shouldn't happen at all, but this happens for Nell-995.
            orig_train_e2_list = self.train_map[(e1, r)]
            temp_train_e2_list = []
            for e2 in orig_train_e2_list:
                if e2 in e2_list:
                    continue
                temp_train_e2_list.append(e2)
            self.train_map[(e1, r)] = temp_train_e2_list
            # also remove (e2, r^-1, e1)
            r_inv = get_inv_relation(r, self.args.dataset)
            temp_map = {}  # map from (e2, r_inv) -> outgoing nodes
            for e2 in e2_list:
                temp_map[(e2, r_inv)] = self.train_map[e2, r_inv]
                temp_list = []
                for e1_dash in self.train_map[e2, r_inv]:
                    if e1_dash == e1:
                        continue
                    else:
                        temp_list.append(e1_dash)
                self.train_map[e2, r_inv] = temp_list

            total_examples += len(e2_list)
            if e1 not in self.entity_vocab:
                # put it back
                self.train_map[(e1, r)] = orig_train_e2_list
                for e2 in e2_list:
                    self.train_map[(e2, r_inv)] = temp_map[(e2, r_inv)]
                continue  # this entity was not seen during train; skip
            
            all_programs = self.get_programs_from_nearest_neighbors(r, self.get_neighbors)
            if all_programs is None or len(all_programs) == 0:
                # put it back
                self.train_map[(e1, r)] = orig_train_e2_list
                for e2 in e2_list:
                    self.train_map[(e2, r_inv)] = temp_map[(e2, r_inv)]
                continue
            for p in all_programs:
                if p[0] == r:
                    continue
                if r not in learnt_programs:
                    learnt_programs[r] = {}
                if p not in learnt_programs[r]:
                    learnt_programs[r][p] = 0
                learnt_programs[r][p] += 1

            if len(all_programs) > 0:
                non_zero_ctr += len(e2_list)

            all_uniq_programs = self.rank_programs(all_programs, r)

            num_programs.append(len(all_uniq_programs))
            # Now execute the program
            answers, not_executed_programs = self.execute_programs(e1, r, all_uniq_programs, self.args.max_branch)

            answers, all_siment = self.aggregate_answers(r, answers)
            if len(answers) > 0:
                _10, _5, _3, _1, rr, mr1 = self.get_hits(answers, e2_list, all_siment, (e1, r))
                hits_10 += _10
                hits_5 += _5
                hits_3 += _3
                hits_1 += _1
                mrr += rr
                mr += mr1
                if self.args.output_per_relation_scores:
                    if r not in per_relation_scores:
                        per_relation_scores[r] = {"hits_1": 0, "hits_3": 0, "hits_5": 0, "hits_10": 0, "mrr": 0}
                        per_relation_query_count[r] = 0
                    per_relation_scores[r]["hits_1"] += _1
                    per_relation_scores[r]["hits_3"] += _3
                    per_relation_scores[r]["hits_5"] += _5
                    per_relation_scores[r]["hits_10"] += _10
                    per_relation_scores[r]["mrr"] += rr
                    per_relation_scores[r]["mr"] += mr1
                    per_relation_query_count[r] += len(e2_list)
            num_answers.append(len(answers))
            # put it back
            self.train_map[(e1, r)] = orig_train_e2_list
            for e2 in e2_list:
                self.train_map[(e2, r_inv)] = temp_map[(e2, r_inv)]

        if self.args.output_per_relation_scores:
            for r, r_scores in per_relation_scores.items():
                r_scores["hits_1"] /= per_relation_query_count[r]
                r_scores["hits_3"] /= per_relation_query_count[r]
                r_scores["hits_5"] /= per_relation_query_count[r]
                r_scores["hits_10"] /= per_relation_query_count[r]
                r_scores["mrr"] /= per_relation_query_count[r]
                r_scores["mr"] /= per_relation_query_count[r]
            out_file_name = os.path.join(self.args.output_dir, "per_relation_scores" + ".json")
            fout = open(out_file_name, "w")
            self.args.logger.info("Writing per-relation scores to {}".format(out_file_name))
            fout.write(json.dumps(per_relation_scores, sort_keys=True, indent=4))
            fout.close()

        self.args.logger.info(
            "Out of {} queries, atleast one program was returned for {} queries".format(total_examples, non_zero_ctr))
        self.args.logger.info("Avg number of programs {:3.2f}".format(np.mean(num_programs)))
        self.args.logger.info("Avg number of answers after executing the programs: {}".format(np.mean(num_answers)))
        self.args.logger.info("Answer in candidate: {}, {}".format(self.in_candidate, self.in_candidate/total_examples))
        self.args.logger.info("{}/{}, Hits@1 {}".format(hits_1, total_examples, hits_1 / total_examples))
        self.args.logger.info("{}/{}, Hits@3 {}".format(hits_3, total_examples, hits_3 / total_examples))
        self.args.logger.info("{}/{}, Hits@5 {}".format(hits_5, total_examples, hits_5 / total_examples))
        self.args.logger.info("{}/{}, Hits@10 {}".format(hits_10, total_examples, hits_10 / total_examples))
        self.args.logger.info("{}/{}, MRR {}".format(mrr, total_examples, mrr / total_examples))
        self.args.logger.info("{}/{}, MR {}".format(mr, total_examples, mr / total_examples))
        self.args.logger.info("Avg number of nn, that do not have the query relation: {}".format(
            np.mean(self.all_zero_ctr)))
        self.args.logger.info("Avg num of returned nearest neighbors: {:2.4f}".format(np.mean(self.all_num_ret_nn)))
        self.args.logger.info("Avg number of programs that do not execute per query: {:2.4f}".format(
            np.mean(self.num_non_executable_programs)))
        if self.args.print_paths:
            for k, v in learnt_programs.items():
                self.args.logger.info("query: {}".format(k))
                self.args.logger.info("=====" * 2)
                for rel, _ in learnt_programs[k].items():
                    self.args.logger.info((rel, learnt_programs[k][rel]))
                self.args.logger.info("=====" * 2)
        if self.args.use_wandb:
            # Log all metrics
            wandb.log({'hits_1': hits_1 / total_examples, 'hits_3': hits_3 / total_examples,
                       'hits_5': hits_5 / total_examples, 'hits_10': hits_10 / total_examples,
                       'mrr': mrr / total_examples, 'total_examples': total_examples, 'non_zero_ctr': non_zero_ctr,
                       'all_zero_ctr': self.all_zero_ctr, 'avg_num_nn': np.mean(self.all_num_ret_nn),
                       'avg_num_prog': np.mean(num_programs), 'avg_num_ans': np.mean(num_answers),
                       'avg_num_failed_prog': np.mean(self.num_non_executable_programs)})

    # Get relation-paths grouped by entity types after the construction of type-specific reasoning schema.
    def get_relation_paths(self, output_filenm=""):
        self.args.logger.info("Get relation-paths of entity types")
        programs_map = {}
        for _, ((e1, r), e2_list) in enumerate(tqdm((self.train_map.items()))):
            c = self.args.cluster_assignments_type[e1]
            if c not in programs_map:
                programs_map[c] = {}
            if r not in programs_map[c]:
                programs_map[c][r] = set()
            all_paths_around_e1 = self.all_paths[e1]
            for nn_ans in e2_list:
                programs = self.get_programs(e1, nn_ans, all_paths_around_e1)
                for p in programs:
                    p = tuple(p)
                    if len(p) == 1:
                        if p[0] == r:  # don't store query relation
                            continue
                    programs_map[c][r].add(p)
        
        if not output_filenm:
            dir_name = os.path.join(self.args.data_dir, "data", self.args.dataset)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            output_filenm = os.path.join(dir_name, "type_relation_paths.pkl")
        self.args.logger.info("Dumping relation-paths of entity types at {}".format(output_filenm))
        with open(output_filenm, "wb") as fout:
            pickle.dump(programs_map, fout)

    # Get precision of each path with regard to a query relation after the construction of cross-type reasoning schema.
    def calc_precision_map(self, cross_output_filenm=""):
        self.args.logger.info("Calculating precision map")
        success_map, total_map = {}, {}  # map from query r to a dict of path and ratio of success
        train_map = [((e1, r), e2_list) for ((e1, r), e2_list) in self.train_map.items()]
        for ((e1, r), e2_list) in tqdm(train_map):
            c = self.args.cluster_assignments_type[e1]
            if c not in success_map:
                success_map[c] = {}
            if c not in total_map:
                total_map[c] = {}
            if r not in success_map[c]:
                success_map[c][r] = {}
            if r not in total_map[c]:
                total_map[c][r] = {}
            for path in self.args.type_relation_paths[c][r]:
                ans = self.execute_one_program(e1, path, depth=0, max_branch=100)
                if len(ans) == 0:
                    continue
                # execute the path get answer
                if path not in success_map[c][r]:
                    success_map[c][r][path] = 0
                if path not in total_map[c][r]:
                    total_map[c][r][path] = 0
                for a in ans:
                    if a in e2_list:
                        success_map[c][r][path] += 1
                    total_map[c][r][path] += 1
        
        s_map_cross = {}
        total_map_cross = {}
        for c, _ in success_map.items():
            for r, _ in success_map[c].items():
                if r not in s_map_cross:
                    s_map_cross[r] = {}
                if r not in total_map_cross:
                    total_map_cross[r] = {}
                for path, s_c in success_map[c][r].items():
                    if path not in s_map_cross[r]:
                        s_map_cross[r][path] = 0
                    if path not in total_map_cross[r]:
                        total_map_cross[r][path] = 0
                    s_map_cross[r][path] = s_map_cross[r][path] + s_c
                    total_map_cross[r][path] = total_map_cross[r][path] + total_map[c][r][path]

        precision_map_cross = {}
        for r in s_map_cross:
            if r not in precision_map_cross:
                precision_map_cross[r] = {}
                for p in s_map_cross[r]:
                    precision_map_cross[r][p] = s_map_cross[r][p] / total_map_cross[r][p]
        self.args.logger.info("Dumping cross precision map at {}".format(cross_output_filenm))
        with open(cross_output_filenm, "wb") as fout:
            pickle.dump(precision_map_cross, fout)
        self.args.logger.info("Dumping cross precision map: Done...")

def main(args):
    dataset = args.dataset
    args.logger.info("==========={}============".format(dataset))
    data_dir = os.path.join(args.data_dir, dataset)
    subgraph_dir = os.path.join("subgraphs", dataset)

    args.train_file = os.path.join(data_dir, "train.txt")
    args.dev_file = os.path.join(data_dir, "dev.txt")
    args.test_file = os.path.join(data_dir, "test.triples") if not args.test_file_name \
            else os.path.join(data_dir, args.test_file_name)
    args.logger.info("test_file:" + args.test_file)

    if args.subgraph_file_name == "":
        args.subgraph_file_name = f"paths_{args.num_paths_to_collect}_{args.max_path_len}hop"
        if args.prevent_loops:
            args.subgraph_file_name += "_no_loops"
        args.subgraph_file_name += ".pkl"

    if os.path.exists(os.path.join(subgraph_dir, args.subgraph_file_name)):
        args.logger.info("Loading subgraph around entities:")
        with open(os.path.join(subgraph_dir, args.subgraph_file_name), "rb") as fin:
            all_paths = pickle.load(fin)
    else:
        args.logger.info("Sampling subgraph around entities:")
        unique_entities = get_unique_entities(args.train_file)
        train_adj_list = create_adj_list(args.train_file)
        all_paths = defaultdict(list)
        for _, e1 in enumerate(tqdm(unique_entities)):
            paths = get_paths(args, train_adj_list, e1, max_len=args.max_path_len)
            if paths is None:
                continue
            all_paths[e1] = paths
        os.makedirs(subgraph_dir, exist_ok=True)
        with open(os.path.join(subgraph_dir, args.subgraph_file_name), "wb") as fout:
            pickle.dump(all_paths, fout)

    entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab = create_vocab(args.train_file)
    args.logger.info("Loading train map")
    train_map = load_data(args.train_file)
    args.logger.info("Loading dev map")
    dev_map = load_data(args.dev_file)
    args.logger.info("Loading test map")
    test_map = load_data(args.test_file)
    eval_map = dev_map
    eval_file = args.dev_file
    if args.test:
        eval_map = test_map
        eval_file = args.test_file

    # get the unique entities in eval set, so that we can calculate similarity in advance.
    eval_entities = get_unique_entities(eval_file)
    eval_vocab, eval_rev_vocab = {}, {}

    e_ctr = 0
    for e in eval_entities:
        eval_vocab[e] = e_ctr
        eval_rev_vocab[e_ctr] = e
        e_ctr += 1

    args.logger.info("=========Config:============")
    args.logger.info(vars(args))

    args.logger.info("Loading combined train/dev/test map for filtered eval")
    all_kg_map = load_data_all_triples(args.train_file, args.dev_file, os.path.join(data_dir, 'test.txt'))
    args.all_kg_map = all_kg_map

    LoGRe_agent = LoGRe(args, train_map, eval_map, entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, eval_vocab, eval_rev_vocab, all_paths)

    
    # Calculate similarity
    adj_mat = read_graph(args.train_file, entity_vocab, rel_vocab)
    adj_mat = np.sqrt(adj_mat)
    l2norm = np.linalg.norm(adj_mat, axis=-1)
    l2norm = l2norm + np.finfo(float).eps
    adj_mat = adj_mat / l2norm.reshape(l2norm.shape[0], 1)
    # Let's put this to GPU
    adj_mat = torch.from_numpy(adj_mat)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    args.logger.info('Using device:'.format(device.__str__()))
    adj_mat = adj_mat.to(device)
    args.logger.info("Calculating ansim matrix")
    ansim = torch.matmul(adj_mat, torch.t(adj_mat))
    LoGRe_agent.set_ansim(ansim.cpu().numpy())
    
    # Cluster entities
    cluster_assignments_type, type2ents = LoGRe_agent.cluster_entities_type(os.path.join(args.data_dir, args.dataset, args.type_file_name))  # dict{e: type}, dict{type: set(e)}
    args.cluster_assignments_type = cluster_assignments_type
    args.type2ents = type2ents

    # Get relation-paths of entity types
    type_relation_paths_filenm = os.path.join(data_dir, "type_relation_paths.pkl")
    if not os.path.exists(type_relation_paths_filenm):
        LoGRe_agent.get_relation_paths(output_filenm=type_relation_paths_filenm)
    args.logger.info("Loading relation-paths of entity types")
    with open(type_relation_paths_filenm, "rb") as fin:
        args.type_relation_paths = pickle.load(fin)

    # Get path scores
    cross_precision_map_filenm = os.path.join(data_dir, "precision_map_cross.pkl")
    if not os.path.exists(cross_precision_map_filenm):
        LoGRe_agent.calc_precision_map(cross_output_filenm=cross_precision_map_filenm)
    args.logger.info("Loading precision map")
    with open(cross_precision_map_filenm, "rb") as fin:
        args.precision_map = pickle.load(fin)
    
    LoGRe_agent.path_reasoning()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect subgraphs around entities")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--dataset", type=str, default="FB15K-237-10")
    parser.add_argument("--out_dir", type=str, default="outputs/")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_file_name", type=str, default='test.triples')
    parser.add_argument("--type_file_name", type=str, default="entity2type.txt")

    parser.add_argument("--subgraph_file_name", type=str, default="")
    parser.add_argument("--max_num_programs", type=int, default=1000)
    parser.add_argument("--num_paths_to_collect", type=int, default=1000)
    parser.add_argument("--max_path_len", type=int, default=3)  #need to tune
    parser.add_argument("--decay_factor", type=float, default=0.95)

    parser.add_argument("--name_of_run", type=str, default="unset")
    parser.add_argument("--output_per_relation_scores", action="store_true")
    parser.add_argument("--print_paths", action="store_true")
    parser.add_argument("--use_wandb", type=int, choices=[0, 1], default=0, help="Set to 1 if using W&B")
    parser.add_argument("--max_branch", type=int, default=1000)
    parser.add_argument("--prevent_loops", type=int, choices=[0, 1], default=1)

    args = parser.parse_args()
    args.logger = Logger("logs", args.name_of_run + "_" + str(args.max_num_programs) + "_" + str(args.num_paths_to_collect) + "_" + str(args.max_path_len) + "_" + str(args.decay_factor)).logger
    args.logger.info('\n\nCOMMAND: %s' % ' '.join(sys.argv))
    if args.use_wandb:
        wandb.init(project='LoGRe')

    if args.name_of_run == "unset":
        args.name_of_run = str(uuid.uuid4())[:8]
    args.output_dir = os.path.join(args.out_dir, args.dataset)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.logger.info(f"Output directory: {args.output_dir}")
    
    hop_factors = [0.0] * 11
    hop_factors[1] = args.decay_factor
    for i in range(2, len(hop_factors)):
        hop_factors[i] = hop_factors[i-1] * args.decay_factor
    args.hop_factors = hop_factors
    main(args)
