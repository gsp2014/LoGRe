import argparse
years = ["concept:dateliteral", "concept:date"]

# For NELL23K, entity types are obtained from entity names, as they are in the form of “concept_type_entity”.
def get_ent_type(train_file):
    ent_type = {}
    with open(train_file, "r") as fin:
        for line_ctr, line in enumerate(fin):
            e1, e2, r = line.strip().split("\t")
            if e1 not in ent_type:
                tmp = e1.strip().split("_")
                if len(tmp) <= 2:
                    print("len(tmp) <= 2", e1)
                if len(tmp) < 2:
                    ent_type[e1] = r + "_"
                else:
                    ent_type[e1] = tmp[0] + ":" + tmp[1]
            if e2 not in ent_type:
                tmp = e2.strip().split("_")
                if len(tmp) <= 2:
                    print("len(tmp) <= 2", e2)
                if len(tmp) < 2:
                    ent_type[e2] = r
                else:
                    ent_type[e2] = tmp[0] + ":" + tmp[1]
    return ent_type

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get entity type for NELL23K.")
    parser.add_argument("--dataset", type=str, default="data/NELL23K/")
    parser.add_argument("--out_type_file", type=str, default="entity2type.txt")
    args = parser.parse_args()
    ent_type = get_ent_type(args.dataset + "train.txt")
    fout = open(args.dataset + args.out_type_file, "w")
    for ent in ent_type:
        if ent_type[ent] in years:
            t = "concept:year"
        else:
            t = ent_type[ent]
        fout.write(ent + "\t" + t + "\n")
    fout.close()
    print("len(ent_type):", len(ent_type))
