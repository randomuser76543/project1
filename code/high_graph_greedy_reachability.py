import json
import os
import sys
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from GraphUtils import GraphUtils
from globals import datasets

sys.setrecursionlimit(5000)


class GreedyReachabilityAnalyzer:
    @staticmethod
    def get_maximum_neighbor_dict(g):
        d = {}
        for v in tqdm(g.vs):
            out_neighbors_and_degs = list((s, g.indegree(s)) for s in g.successors(v))
            max_neigh, max_neigh_degree = max(out_neighbors_and_degs, key=lambda x: x[1]) if len(
                out_neighbors_and_degs) > 0 else (None, None)
            d[v["name"]] = None if not out_neighbors_and_degs or max_neigh_degree <= v.indegree() else max_neigh
        return d

    @staticmethod
    def max_reachable_greedy_rec(g, v, max_neighbour_dict, max_reachable_dict={}):
        def increment_tuple(node, dist):
            return node, dist + 1

        # memoization
        if v in max_reachable_dict:
            return max_reachable_dict[v]

        max_neighbour = max_neighbour_dict[v]
        # either reached a local maximum, or need to recursively search for it.
        max_reachable_dict[v] = ((v, 0) if max_neighbour is None
                                 else increment_tuple(*GreedyReachabilityAnalyzer.max_reachable_greedy_rec(
            g, max_neighbour, max_neighbour_dict, max_reachable_dict)))

        return max_reachable_dict[v]

    @staticmethod
    def get_reachability_dict(g):
        reachability_dict = {}
        print("Building max-neighbour dictionary")
        max_neigh_dict = GreedyReachabilityAnalyzer.get_maximum_neighbor_dict(g)
        print("Building reachability dictionary")
        for u in tqdm(g.vs):
            GreedyReachabilityAnalyzer.max_reachable_greedy_rec(g=g, v=u["name"], max_neighbour_dict=max_neigh_dict,
                                                                max_reachable_dict=reachability_dict)
        return reachability_dict

    @staticmethod
    def get_reachable_high_graph_dist(g, in_degrees, min_follower_out_degree=20, min_high_nodes_frac=0.0001,
                                      max_high_nodes_frac=.0101, step=0.0005):
        high_graph_fracs = np.arange(min_high_nodes_frac, max_high_nodes_frac, step)
        print("Obtaining reachability dictionary")
        reachability_dict = GreedyReachabilityAnalyzer.get_reachability_dict(g)
        node_to_reaching_dict = defaultdict(set)
        for u, (v, d) in reachability_dict.items():
            if g.outdegree(u) >= min_follower_out_degree:
                node_to_reaching_dict[v].add(u)

        reachable_nodes = set(node_to_reaching_dict.keys())
        print("Computing fraction of nodes greedy reaching high nodes")
        num_covered_list = []
        for frac in tqdm(high_graph_fracs):
            high_nodes = set([u for u, _ in in_degrees[:int(frac * g.vcount())]])
            reachable_high_nodes = high_nodes.intersection(reachable_nodes).union()
            if len(reachable_high_nodes) == 0:
                num_covered_list.append(0)
            else:
                num_covered_list.append(len(set.union(*[node_to_reaching_dict[v] for v in reachable_high_nodes])))
        num_eligible_covered = len([u for u in g.vs if g.outdegree(u) >= min_follower_out_degree])
        return high_graph_fracs, np.array(num_covered_list) / num_eligible_covered

    @staticmethod
    def plot_max_reachable_fractions(paths_sep_and_titles, min_follower_out_degree=20, results_file=None,
                                     output_file=None):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.switch_backend('agg')

        plot_values = {}
        if results_file and os.path.isfile(results_file):
            with open(results_file, "r") as f:
                plot_values = json.load(f)

        for file_path, sep, title, is_directed in paths_sep_and_titles:
            print("Analyzing the reachabilities of the high nodes: ", title)
            if title not in plot_values:
                g, in_degrees = GraphUtils.load_graph(file_path=file_path, sep=sep, directed=is_directed)
                high_graph_fracs, reachable_fracs = GreedyReachabilityAnalyzer.get_reachable_high_graph_dist(g,
                                                                                                             in_degrees,
                                                                                                             min_follower_out_degree=min_follower_out_degree)
                plot_values[title] = high_graph_fracs.tolist(), reachable_fracs.tolist()

        for _, _, title, _ in paths_sep_and_titles:
            high_graph_fracs, reachable_fracs = plot_values[title]
            plt.plot(high_graph_fracs, reachable_fracs, label=title)

        plt.legend(loc='lower right', ncol=2)
        plt.xlabel("Number of nodes in the high graph as fraction of $n$")
        plt.ylabel("Fraction of $V$ reaching high nodes")
        # plt.title(f"Fraction of nodes of out-degree $\geq 20$ reaching high nodes via locally greedy traversal")
        plt.xlim(0.0001, 0.0099)
        plt.locator_params(axis='x', nbins=12)
        if results_file:
            with open(results_file, "w") as f:
                json.dump(plot_values, f, sort_keys=True, indent=4, separators=(',', ': '))

        if output_file:
            plt.savefig(output_file, bbox_inches='tight')


if __name__ == "__main__":
    sys.setrecursionlimit(17000)
    min_out_degree = 20
    res_file = f"../results/high-nodes-greedy-reachability.json"
    print(f"Result file: {res_file}")
    GreedyReachabilityAnalyzer.plot_max_reachable_fractions(datasets, results_file=res_file,
                                                            output_file=f"../figures/greedy-reachability-min-out-deg-{min_out_degree}.pdf",
                                                            min_follower_out_degree=20)
