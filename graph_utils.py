import json
import os

class GraphUtils(object):
    def __init__(self):
        with open('./Data/user_data.json','r') as f:
            self.user_data = json.load(f)

        with open('./Data/rt_data.json','r') as f:
            self.rt_data = json.load(f)

        self.combined_data = dict()

        for key,val in self.user_data.items():
            self.combined_data[key] = val + self.rt_data[key]

    def generate_edgelist(self):
        fout = './Data/graph.edgelist'
        fo = open(fout, 'w')
        for node1,nodes in self.combined_data.items():
            for node2 in nodes:
                fo.write(str(node1) + ' ' + str(node2))
                fo.write('\n')
        fo.close()

    def generate_node2vec_embeddings(self):
        os.system("python ./node2vec/src/main.py --input ./Data/graph.edgelist --output Data/user_embeddings.emd")

if __name__ == '__main__':
    graph_utils = GraphUtils()
    graph_utils.generate_edgelist()
    graph_utils.generate_node2vec_embeddings()
