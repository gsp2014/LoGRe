import argparse
import numpy as np
skip_types = ["/common/topic"]

# Get entity set from the training set
def get_ent_set(train_file):
    ent_set = set()
    with open(train_file, "r") as fin:
        for line_ctr, line in enumerate(fin):
            e1, e2, r = line.strip().split("\t")
            ent_set.add(e1)
            ent_set.add(e2)
    return ent_set

# Read entity types of FB15K from entity2type_FB15K.txt
def get_types_FB15K(type_file):
    ent_types = {}
    with open(type_file, "r") as fin:
        for line_ctr, line in enumerate(fin):
            tmp = line.strip().split("\t")
            if tmp[0] not in ent_types:
                ent_types[tmp[0]] = []
            for i in range(1, len(tmp)):
                if tmp[i] in skip_types:
                    continue
                if tmp[i] in ent_types[tmp[0]]:
                    print("duplicate type:", tmp[i], "for entity", tmp[0])
                else:
                    ent_types[tmp[0]].append(tmp[i])
    return ent_types

# Filter to retain only the entities in ent_set and their corresponding types.
def filter_types(type_file, ent_set):
    ent_types = get_types_FB15K(type_file)
    filtered_ent_types = {}
    for ent in ent_set:
        if ent in ent_types:
            filtered_ent_types[ent] = ent_types[ent]
        else:
            print(ent)
    return filtered_ent_types

# Sort entity types by their frequency of occurrence.
def sort_types_by_counts(ent_types):
    type_count = {}
    for ent in ent_types:
        for t in ent_types[ent]:
            if t not in type_count:
                type_count[t] = 0
            type_count[t] = type_count[t] + 1
    sorted_type_count = sorted(type_count.items(), key=lambda x:x[1], reverse=True)
    sorted_type_list = []
    for i in sorted_type_count:
        sorted_type_list.append(i[0])
    return sorted_type_list

# Select the most frequent type as the entity type.
def get_final_type(ent_set, ent_type, sorted_type_list):
    final_ent_type = {}
    not_type_entity = set()
    for ent in ent_set:
        indexes = []
        if ent not in ent_type:
            not_type_entity.add(ent)
            final_ent_type[ent] = "others"  # If no type is available for an entity, it is assigned to the type “others”
            continue
        for t in ent_type[ent]:
            indexes.append(sorted_type_list.index(t))
        min_val = np.array(indexes).min()
        final_ent_type[ent] = ent_type[ent][indexes.index(min_val)]
    return final_ent_type, not_type_entity

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get entity type for FB15K-237-x.")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--dataset", type=str, default="FB15K-237-10")
    parser.add_argument("--in_type_file", type=str, default="entity2type_FB15K.txt")
    parser.add_argument("--out_type_file", type=str, default="entity2type.txt")
    
    args = parser.parse_args()
    ent_set = get_ent_set(args.data_dir+args.dataset+"/" + "train.txt")
    filtered_ent_types = filter_types(args.data_dir + args.in_type_file, ent_set)
    sorted_type_list = sort_types_by_counts(filtered_ent_types)
    final_ent_type, not_type_entity = get_final_type(ent_set, filtered_ent_types, sorted_type_list)
    fout = open(args.data_dir + args.dataset + "/" + args.out_type_file, "w")
    for ent in final_ent_type:
        fout.write(ent+"\t"+final_ent_type[ent]+"\n")
    fout.close()
    print("len(final_ent_type):", len(final_ent_type))
    print("not_type_entity:", not_type_entity, len(not_type_entity))
