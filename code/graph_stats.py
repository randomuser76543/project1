import json
import os

import numpy as np
from tqdm import tqdm as tqdm

from GraphUtils import GraphUtils
from globals import datasets


class GraphStats:

    @staticmethod
    def filter_by_out_degree(g, min_out_degree=10):
        vertices = [u for u, out_deg in enumerate(g.outdegree()) if out_deg >= min_out_degree]
        return g.subgraph(vertices)

    @staticmethod
    def get_avg_high_graph_neighbors(g, in_degrees, high_graph_frac, min_outdegree):
        high_nodes = set([u for u, _ in in_degrees[:int(g.vcount() * high_graph_frac)]])
        return np.array([len(set(g.neighbors(u)).intersection(high_nodes)) for u in tqdm(g.vs) if
                         g.outdegree(u) >= min_outdegree]).mean()

    @staticmethod
    def get_robust_power_law_exp(g, min_deg=10):
        degrees = np.array(g.degree())
        degrees_filtered = np.extract(degrees >= min_deg, degrees)
        return 1 + len(degrees_filtered) * 1 / np.sum(np.log(degrees_filtered / min_deg))

    @staticmethod
    def analyze_graphs(datasets, results_file=None, min_out_deg_list=[0, 10, 20],
                       high_graph_fracs=[0.0001, 0.001, 0.01]):

        stats_values = {}
        if results_file and os.path.isfile(results_file):
            with open(results_file, "r") as f:
                stats_values = json.load(f)

        for file_path, sep, title, is_directed in datasets:
            if title not in stats_values:
                stats_values[title] = {}
                print(f"Loading graph {title}")
                g, in_degrees = GraphUtils.load_graph(file_path=file_path, sep=sep, directed=is_directed)
                for min_out_deg in min_out_deg_list:
                    print(f"Getting stats for minimum out-degree: {min_out_deg}")
                    key = f"min_out_deg={min_out_deg}"
                    stats_values[title][key] = {}
                    print(f"Obtaining valid subgraph for min out degree {title}")
                    valid_subgraphx = GraphStats.filter_by_out_degree(g=g, min_out_degree=min_out_deg)
                    stats_values[title][key]["n"] = valid_subgraphx.vcount()
                    stats_values[title][key]["m"] = valid_subgraphx.ecount()
                    stats_values[title][key]["avg_degree"] = 2 * valid_subgraphx.ecount() / valid_subgraphx.vcount()
                    print("Obtainging average number of neighbors")
                    for frac in high_graph_fracs:
                        print(f"High graph fraction={frac}")
                        stats_values[title][key][
                            f"avg_num_high_graph_neighbors_{frac}"] = GraphStats.get_avg_high_graph_neighbors(g,
                                                                                                              in_degrees,
                                                                                                              frac,
                                                                                                              min_out_deg)

        if results_file:
            with open(results_file, "w") as f:
                json.dump(stats_values, f, sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    GraphStats.analyze_graphs(datasets=datasets, results_file=f"../results/graph-stats.json")
