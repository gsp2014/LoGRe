import argparse
import numpy as np
from mongo_util import MongoUtil

type_pid = "P31"
ignore_eids = []
ignore_ename = []

def get_ent_set(train_file):
    ent_set = set()
    with open(train_file, "r") as fin:
        for line_ctr, line in enumerate(fin):
            e1, e2, r = line.strip().split("\t")
            ent_set.add(e1)
            ent_set.add(e2)
    return ent_set

# Given entity id, get entity name from Wikidata MongoDB.
def get_name(table, eid):
    names = mongoUtil.findAll(table, condition={"Entity_id": eid}, fields={"Entity_name": 1})
    names = list(names)
    if len(names) == 0:
        return ""
    
    flag = 0
    for i in range(len(names)):
        if names[i]["Entity_name"] == {}:
            continue
        if "en" in names[i]["Entity_name"]:
            flag = 1
            break
    if flag == 0:
        return ""
    else:
        return names[i]["Entity_name"]["en"]

# Check if the given Qid corresponds to a valid entity.
def check_Qent(Qid):
    if Qid in ignore_eids:
        return 0
    if Qid.startswith("P"):
        return 0
    tname = get_name(args.entity_collection, Qid)
    if tname == "":
        return 0
    for i in ignore_ename:
        if tname == i or tname.startswith(i+' ') or tname.endswith(' '+i) or tname.find(' '+i+' ') != -1:
            return 0
    return 1

# Process tail name.
def process_tail(tail):
    if tail["type"] not in ["no_value", "uncertain_value", "globecoordinate", "monolingualtext"]:  #"no value", "unknown value"
        if tail["type"] == "quantity":
            tail_n = tail["value"]["amount"]
            if tail["value"]["unit"] != "1":
                unit = tail["value"]["unit"]
                if type(unit) != str:
                    unit = str(unit)
                tail_n = tail_n + "~" + unit
        elif tail["type"] in ["sitelinks", "string"]:
            tail_n = tail["value"]
        elif tail["type"] == "time":
            tail_n = tail["value"]["time"]
            if "timezone" in tail["value"]:
                tail_n = tail_n + "~timezone" + str(tail["value"]["timezone"])
        elif tail["type"] == "wikibase-entityid":
            tail_n = get_name(args.entity_collection, tail["value"]["id"])
        else:
            print(tail["type"]+" error!")
            tail_n = ""
    else:
        tail_n = ""
    return tail_n

def get_final_tail(tail):
    if tail["type"] in ["wikibase-entityid"]:
        if check_Qent(tail["value"]["id"]) == 0:
            return ""
    tail_n = process_tail(tail)
    return tail_n

# Get entity types from Wikidata MongoDB.
def get_types(ent_set, mongoUtil):
    ent_types = {}
    for ent in ent_set:
        types = mongoUtil.findAll(args.relation_collection, condition={"$and":[{"Head_id": ent}, {"Relation_id": type_pid}]}, fields={"Tail": 1, "Qualifier": 1})
        types = list(types)
        if len(types) == 0:
            continue
        ent_types[ent] = []
        for one in types:
            tail_n = get_final_tail(one["Tail"])
            ent_types[ent].append(tail_n)
    return ent_types

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
    parser = argparse.ArgumentParser(description="Get entity type for WD-singer.")
    parser.add_argument("--dataset_data_dir", type=str, default="data/WD-singer/")
    parser.add_argument("--type_file_name", type=str, default="entity2type.txt")
    parser.add_argument('-host', dest='host', default='xx.xx.xx.xx', type=str, help='The MongoDB host for Wikidata.')
    parser.add_argument('-port', dest='port', default=27017, type=int, help='The MongoDB port for Wikidata.')
    parser.add_argument('-user', dest='user', default='wikidata', type=str, help='user.')
    parser.add_argument('-passwd', dest='passwd', default='111111', type=str, help='passwd.')
    parser.add_argument('-authenticate_dbName', dest='authenticate_dbName', default='wikidata', type=str, help='authenticate_dbName.')
    parser.add_argument('-find_dbName', dest='find_dbName', default='wikidata', type=str, help='find_dbName.')
    parser.add_argument('-entity_collection', dest='entity_collection', default='Entity', type=str, help='entity table.')
    parser.add_argument('-property_collection', dest='property_collection', default='Property', type=str, help='property table.')
    parser.add_argument('-relation_collection', dest='relation_collection', default='Relation', type=str, help='relation table')
    args = parser.parse_args()
    mongoUtil = MongoUtil(args.host, args.port, args.user, args.passwd, args.authenticate_dbName, args.find_dbName)

    ent_set = get_ent_set(args.dataset_data_dir + "train.txt")
    ent_types = get_types(ent_set, mongoUtil)
    sorted_type_list = sort_types_by_counts(ent_types)
    final_ent_type, not_type_entity = get_final_type(ent_set, ent_types, sorted_type_list)
    fout = open(args.dataset_data_dir + args.type_file_name, "w")
    for ent in final_ent_type:
        fout.write(ent+"\t"+final_ent_type[ent]+"\n")
    fout.close()
    print("len(final_ent_type):", len(final_ent_type))
    print("not_type_entity:", not_type_entity, len(not_type_entity))
