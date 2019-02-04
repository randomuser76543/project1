import json
import os

import numpy as np
from functools import partial
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm

import globals
from GraphUtils import GraphUtils
from globals import datasets


class HighGraphSCCAnalyzer:
    @staticmethod
    def get_num_cc_in_high_graph(g, in_degrees, num_high_nodes):
        return len(GraphUtils.get_high_graph(g, in_degrees, num_high_nodes=num_high_nodes).clusters(mode="WEAK").sizes())

    @staticmethod
    def get_num_cc_dist(g, in_degrees, min_num_high_frac=.0001, max_num_high_nodes_frac=.0101, step=0.0001):
        high_graph_fracs = np.arange(min_num_high_frac, max_num_high_nodes_frac, step)
        get_num_cc_for_num_high_nodes = partial(HighGraphSCCAnalyzer.get_num_cc_in_high_graph, g, in_degrees)
        n_cc_list = map(get_num_cc_for_num_high_nodes, (high_graph_fracs * g.vcount()).astype(int))
        return high_graph_fracs, list(n_cc_list)

    @staticmethod
    def analyze_high_graph_connectedness(file_path, sep, title, min_num_high_frac=.0001, max_num_high_nodes=.0101, step=0.0001):
        g, in_degrees = GraphUtils.load_graph(file_path, sep)
        high_graph_fracs, n_scc_list = HighGraphSCCAnalyzer.get_num_cc_dist(g, in_degrees, min_num_high_frac, max_num_high_nodes, step)
        plt.plot(high_graph_fracs, n_scc_list)
        plt.title(title)
        plt.xlabel("Number of high nodes, as fraction of $n$")
        plt.ylabel("# of CC")

    @staticmethod
    def get_max_cc_coverage(g, in_degrees, num_high_nodes, min_follower_out_degree=10, followers_dict={}):
        largest_component = GraphUtils.get_largest_cc_in_high_graph(g, in_degrees, num_high_nodes)
        followers_total = set()

        for v in largest_component:
            if v not in followers_dict:
                followers_dict[v] = set([u for u in g.predecessors(v)
                                         if g.outdegree(u) >= min_follower_out_degree])
            followers_total |= set(followers_dict[v])

        return followers_total

    @staticmethod
    def get_max_cc_coverage_dist(g, in_degrees, min_follower_out_degree=15, min_high_nodes_frac=0.0001, max_high_nodes_frac=.0101, step=0.0002):
        high_graph_fracs = np.arange(min_high_nodes_frac, max_high_nodes_frac, step)

        num_covered_list = []
        in_neighbors_dict = {}
        num_nodes = len([u['name'] for u in g.vs if u.outdegree() >= min_follower_out_degree])
        for frac in tqdm(high_graph_fracs):
            covered = HighGraphSCCAnalyzer.get_max_cc_coverage(g, in_degrees, int(frac * g.vcount()), min_follower_out_degree, in_neighbors_dict)
            num_covered_list.append(len(covered) / num_nodes)
        return high_graph_fracs, np.array(num_covered_list)

    @staticmethod
    def plot_max_cc_coverages(datasets, results_file=None, min_out_deg=15, output_file=None):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.switch_backend('agg')

        plot_values = {}
        if results_file and os.path.isfile(results_file):
            with open(results_file, "r") as f:
                plot_values = json.load(f)

        for file_path, sep, title, is_directed in datasets:
            if title not in plot_values:
                print("Analyzing the coverage of the largest CC: ", title)
                g, in_degrees = GraphUtils.load_graph(file_path=file_path, sep=sep, directed=is_directed)
                globals.graph = g
                high_graph_fracs, coverage_fracs = HighGraphSCCAnalyzer.get_max_cc_coverage_dist(g, in_degrees, min_follower_out_degree=min_out_deg)
                plot_values[title] = high_graph_fracs.tolist(), coverage_fracs.tolist()

            high_graph_fracs, coverage_fracs = plot_values[title]
            plt.plot(high_graph_fracs, coverage_fracs, label=title)

        print("Generating figure: ", output_file)
        plt.legend(loc='lower right', ncol=2)
        plt.xlabel("Number of nodes in the high graph as fraction of $n$")
        plt.ylabel("Fraction covered")
        plt.xlim(0.000, 0.0099)
        plt.locator_params(axis='x', nbins=14)

        # plt.title(f"Coverage of largest CC in the high graph. nodes. Minimum follower out-deg.: {min_out_deg}")
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
        if results_file:
            with open(results_file, "w") as f:
                json.dump(plot_values, f, sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    min_out_deg = 20
    HighGraphSCCAnalyzer.plot_max_cc_coverages(datasets,
                                               results_file=f"../results/max_cc_coverage-min-out-{min_out_deg}.json",
                                               output_file=f"../figures/max_cc_coverage-min-out-{min_out_deg}.pdf",
                                               min_out_deg=min_out_deg)
