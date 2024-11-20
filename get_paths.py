from tqdm import tqdm
from collections import defaultdict
import os
import numpy as np
from data_utils import get_unique_entities, create_adj_list
import time
import pickle
import argparse
import wandb
from log import Logger

# Given start node, randomly collect a maximum number of args.num_paths_to_collect paths and the hop of each path is restricted to a maximum of max_len.
def get_paths(args, train_adj_list, start_node, max_len=3):
    all_paths = set()
    for k in range(args.num_paths_to_collect):
        path = []
        curr_node = start_node
        entities_on_path = set([start_node])
        for l in range(max_len):
            outgoing_edges = train_adj_list[curr_node]
            if args.prevent_loops:
                # Prevent loops
                temp = []
                for oe in outgoing_edges:
                    if oe[1] in entities_on_path:
                        continue
                    else:
                        temp.append(oe)
                outgoing_edges = temp
            if len(outgoing_edges) == 0:
                break
            # pick one at random
            out_edge_idx = np.random.choice(range(len(outgoing_edges)))
            out_edge = outgoing_edges[out_edge_idx]
            path.append(out_edge)
            curr_node = out_edge[1]  # assign curr_node as the node of the selected edge
            entities_on_path.add(out_edge[1])
        all_paths.add(tuple(path))
    return all_paths

def main(args):
    args.logger.info("============={}================".format(args.dataset))
    data_dir = os.path.join(args.data_dir, args.dataset)
    out_dir = os.path.join("subgraphs", args.dataset)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args.prevent_loops = (args.prevent_loops == 1)
    args.logger.info(vars(args))

    train_file = os.path.join(data_dir, "train.txt")
    unique_entities = get_unique_entities(train_file)
    train_adj_list = create_adj_list(train_file)
    st_time = time.time()
    paths_map = defaultdict(list)
    for ctr, e1 in enumerate(tqdm(unique_entities)):
        paths = get_paths(args, train_adj_list, e1, args.max_len)
        if paths is None:
            continue
        paths_map[e1] = paths
        if args.use_wandb and ctr % 100 == 0:
            wandb.log({"progress": ctr / len(unique_entities)})

    args.logger.info("Took {} seconds to collect paths for {} entities".format(time.time() - st_time, len(paths_map)))

    out_file_name = "paths_"+str(args.num_paths_to_collect)+"_"+str(args.max_len)+"hop"
    if args.prevent_loops:
        out_file_name += "_no_loops"
    out_file_name += ".pkl"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    fout = open(os.path.join(out_dir, out_file_name), "wb")
    args.logger.info("Saving at {}".format(out_file_name))
    pickle.dump(paths_map, fout)
    fout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect subgraphs around entities")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--dataset", type=str, default="FB15K-237-10")
    parser.add_argument("--num_paths_to_collect", type=int, default=1000)
    parser.add_argument("--max_len", type=int, default=3)
    parser.add_argument("--prevent_loops", type=int, choices=[0, 1], default=1, help="prevent sampling of looped paths")
    parser.add_argument("--use_wandb", type=int, choices=[0, 1], default=0, help="Set to 1 if using W&B")
    args = parser.parse_args()
    args.logger = Logger("logs", "get_paths_" + str(args.num_paths_to_collect) + "_" + str(args.max_len) + "hop_" + str(args.prevent_loops) + "loops").logger
    if args.use_wandb:
        wandb.init(project='collect-paths')

    main(args)
