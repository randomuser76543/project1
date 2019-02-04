import json
import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm

from GraphUtils import GraphUtils
from globals import datasets


class MaxCCHighGraphFractionAnalysis:
    @staticmethod
    def get_max_cc_fraction_dist(g, in_degrees, min_high_nodes_frac=0.0001, max_high_nodes_frac=.0101, step=0.0002):
        high_graph_fracs = np.arange(min_high_nodes_frac, max_high_nodes_frac, step)

        max_cc_fracs = []

        for frac in tqdm(high_graph_fracs):
            num_high_nodes = int(frac * g.vcount())
            max_cc = GraphUtils.get_largest_cc_in_high_graph(g, in_degrees, num_high_nodes)
            max_cc_fracs.append(len(max_cc) / num_high_nodes)
        return high_graph_fracs, np.array(max_cc_fracs)

    @staticmethod
    def plot_max_cc_fractions(paths_sep_and_titles, results_file=None, output_file=None):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.switch_backend('agg')

        plot_values = {}
        if results_file and os.path.isfile(results_file):
            with open(results_file, "r") as f:
                plot_values = json.load(f)

        for file_path, sep, title, is_directed in paths_sep_and_titles:

            print("Analyzing the coverage of the largest CC: ", title)
            if title not in plot_values:
                g, in_degrees = GraphUtils.load_graph(file_path=file_path, sep=sep, directed=is_directed)
                high_graph_fracs, max_cc_fracs = MaxCCHighGraphFractionAnalysis.get_max_cc_fraction_dist(g, in_degrees)
                plot_values[title] = high_graph_fracs.tolist(), max_cc_fracs.tolist()

            high_graph_fracs, max_cc_fracs = plot_values[title]
            plt.plot(high_graph_fracs, max_cc_fracs, label=title)
        plt.legend(loc='lower right', ncol=2)
        plt.xlabel("Number of nodes in the high graph as fraction of $n$")
        plt.ylabel("Fraction of high graph")
        plt.xlim(0.000, 0.0099)
        plt.locator_params(axis='x', nbins=12)
        # plt.title(f"Fraction of the high graph taken-up by the largest CC")
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
        if results_file:
            with open(results_file, "w") as f:
                json.dump(plot_values, f, sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    min_out_deg = 20
    MaxCCHighGraphFractionAnalysis.plot_max_cc_fractions(datasets,
                                                         results_file=f"../results/max_cc_fraction-min-out-{min_out_deg}.json",
                                                         output_file=f"../figures/max_cc_fraction-min-out-{min_out_deg}.pdf", )
