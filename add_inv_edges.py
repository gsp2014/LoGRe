import argparse

def add_inv_edges(dataset):
    fs = ["train.triples", "dev.triples", "test.triples"]
    for i in range(len(fs)):
        fout = open(dataset + fs[i][:-7] + "txt", "w")
        with open(dataset + fs[i], "r") as fin:
            for line_ctr, line in enumerate(fin):
                e1, e2, r = line.strip().split("\t")
                fout.write("{}\t{}\t{}\n".format(e1, e2, r))
                fout.write("{}\t{}\t{}\n".format(e2, e1, r+"_inv"))
        fout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Add inverse edges.")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--dataset", type=str, default="FB15K-237-10")
    args = parser.parse_args()
    """ datasets = ["FB15K-237-10", "FB15K-237-20", "FB15K-237-50", "NELL23K", "WD-singer"]
    #datasets = ["WD-singer"]
    for dataset in datasets:
        add_inv_edges(args.data_dir + dataset + "/") """
    
    add_inv_edges(args.data_dir + args.dataset + "/")