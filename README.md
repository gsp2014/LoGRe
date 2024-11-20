# LoGRe
This repository provides the code for the paper "Look Globally and Reason: Two-stage Path Reasoning over Sparse Knowledge Graphs", published in CIKM 2024.

## Prepare data
### Add inverse relations and facts

    python add_inv_edges.py --dataset [FB15K-237-10/FB15K-237-20/FB15K-237-50/NELL23K/WD-singer]

### Get paths
For each entity, randomly collect no more than num_paths_to_collect paths from the training data. To avoid the time-consuming collection of excessively long paths that may not contribute to the reasoning, the hop of each path is no more than max_len:

    python get_paths.py --dataset FB15K-237-10 --num_paths_to_collect 20000 --max_len 6
    python get_paths.py --dataset FB15K-237-20 --num_paths_to_collect 5000 --max_len 5
    python get_paths.py --dataset FB15K-237-50 --num_paths_to_collect 1000 --max_len 4
    python get_paths.py --dataset NELL23K --num_paths_to_collect 10000 --max_len 6
    python get_paths.py --dataset WD-singer --num_paths_to_collect 20000 --max_len 6

### Get entity types
- For the FB15K-237 series, retrieve entity
types from [FB15K-237](https://github.com/thunlp/TKRL). 
- For NELL23K, entity types are obtained
from entity names, as they are in the form of “concept_type_entity”.
- For WD-singer, obtain entity types from Wikidata via the
tail entities of relation “*instance of*”.

When there are multiple types, select the most frequent one as the type to mitigate the impact of noisy data. If no type is available for an entity, it is assigned to the type “others”.

    python get_entity_type_FB15K.py --dataset FB15K-237-10
    python get_entity_type_FB15K.py --dataset FB15K-237-20
    python get_entity_type_FB15K.py --dataset FB15K-237-50
    python get_entity_type_NELL23K.py
    python get_entity_type_WD-singer.py

## Reproduce the results presented in the paper

    CUDA_VISIBLE_DEVICES=0 python LoGRe.py --dataset FB15K-237-10 --test --subgraph_file_name paths_20000_6hop_no_loops.pkl --max_num_programs 1000 --num_paths_to_collect 20000 --max_path_len 6 --decay_factor 0.95 --name_of_run FB15K-237-10
    CUDA_VISIBLE_DEVICES=0 python LoGRe.py --dataset FB15K-237-20 --test --subgraph_file_name paths_5000_5hop_no_loops.pkl --max_num_programs 500 --num_paths_to_collect 5000 --max_path_len 5 --decay_factor 0.6 --name_of_run FB15K-237-20
    CUDA_VISIBLE_DEVICES=0 python LoGRe.py --dataset FB15K-237-50 --test --subgraph_file_name paths_1000_4hop_no_loops.pkl --max_num_programs 100 --num_paths_to_collect 1000 --max_path_len 4 --decay_factor 0.8 --name_of_run FB15K-237-50
    CUDA_VISIBLE_DEVICES=0 python LoGRe.py --dataset NELL23K --test --subgraph_file_name paths_10000_6hop_no_loops.pkl --max_num_programs 100 --num_paths_to_collect 10000 --max_path_len 6 --decay_factor 0.5 --name_of_run NELL23K
    CUDA_VISIBLE_DEVICES=0 python LoGRe.py --dataset WD-singer --test --subgraph_file_name paths_20000_6hop_no_loops.pkl --max_num_programs 100 --num_paths_to_collect 20000 --max_path_len 6 --decay_factor 0.2 --name_of_run WD-singer

## Citation
If you found this codebase or our work useful, please cite:

    @inproceedings{LoGRe,
      title={Look Globally and Reason: Two-stage Path Reasoning over Sparse Knowledge Graphs},
      author={Guan, Saiping and Wei, Jiyao and Jin, Xiaolong and Guo, Jiafeng and Cheng, Xueqi},
      booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM 2024)},
      year={2024},
      pages={695-705}
    }

## Related work
[Probabilistic Case-based Reasoning for Open-World Knowledge Graph Completion](https://github.com/ameyagodbole/Prob-CBR)
