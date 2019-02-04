import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from globals import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


class HighGraphCoverageAnalyzer:

    @staticmethod
    def get_num_of_nodes(adj_df):
        followees = set(adj_df.reset_index()["v"].values)
        followers = set.union(*adj_df[~adj_df["followers"].isna()]["followers"])
        return len(followees | followers)

    @staticmethod
    def load_graph(file_path, sep=" ", directed=True):
        print("Reading CSV file")
        df = pd.read_csv(file_path, sep=sep, header=None, names=["source", "target"], comment="#")
        if not directed:
            df_reversed = df[["target", "source"]]
            df_reversed.columns = ["source", "target"]
            df = pd.concat([df, df_reversed], sort=False).drop_duplicates()
        print("Obtaining lists of followers")
        followers = df.groupby("target")["source"].apply(set).to_frame().rename(columns={"source": "followers"})
        followers.index.name = "v"
        print("Computing in-degrees")
        followers["in_deg"] = followers["followers"].apply(len).astype("int64")

        print("Computing out-degrees")
        out_degs = df.groupby("source").size().to_frame().rename(columns={0: "out_deg"})
        out_degs.index.name = "v"

        summary_df = followers.merge(out_degs, how="outer", on="v")
        summary_df.in_deg.fillna(0, inplace=True)
        summary_df.out_deg.fillna(0, inplace=True)
        print("Sorting by in-degrees")
        summary_df.sort_values(by="in_deg", ascending=False, inplace=True)
        print("Done. Returning results.")
        return summary_df, HighGraphCoverageAnalyzer.get_num_of_nodes(summary_df)

    @staticmethod
    def get_high_graph_coverage_incremental(summary_df, max_high_node_frac=0.01, min_out_deg=1):

        nodes = set(summary_df[summary_df["out_deg"] >= min_out_deg].reset_index()["v"])
        followers = set()
        cum_covered = []
        max_high_nodes = int(len(summary_df) * max_high_node_frac)
        for _, r in summary_df.iloc[:max_high_nodes].reset_index().iterrows():
            followers |= r.followers & nodes
            cum_covered.append(len(followers))
        coverage_fracs = np.array(cum_covered) / len(nodes)
        high_graph_fracs = np.array(range(1, len(cum_covered) + 1)) / len(summary_df)

        return coverage_fracs, high_graph_fracs

    @staticmethod
    def get_high_graph_coverage(summary_df, high_graph_fracs=[0.0001, 0.0005, 0.001, 0.004, 0.006, 0.008, 0.01],
                                min_out_deg=20):

        nodes = set(summary_df[summary_df["out_deg"] >= min_out_deg].reset_index()["v"])
        followers = set()
        cum_covered = []
        for frac in tqdm(high_graph_fracs):
            max_high_nodes = int(frac * len(summary_df))
            followers = set.union(*summary_df.iloc[:max_high_nodes]["followers"]).intersection(nodes)
            cum_covered.append(1. * len(followers) / len(nodes))

        return high_graph_fracs, cum_covered

    @staticmethod
    def plot_coverages(datasets, results_file=None, min_out_deg=20, output_file=None):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.switch_backend('agg')

        plot_values = {}
        if results_file and os.path.isfile(results_file):
            with open(results_file, "r") as f:
                plot_values = json.load(f)

        for file_path, sep, title, is_directed in datasets:
            if title not in plot_values:
                print("Analyzing the coverage of the high subgraph: ", title)
                df, n = HighGraphCoverageAnalyzer.load_graph(file_path=file_path, sep=sep, directed=is_directed)
                plot_values[title] = HighGraphCoverageAnalyzer.get_high_graph_coverage(df, min_out_deg=min_out_deg)

            high_graph_fracs, coverage_fracs = plot_values[title]
            plt.plot(high_graph_fracs, coverage_fracs, label=title)

        print("Generating figure: ", output_file)
        plt.legend(loc='lower right', ncol=2)
        plt.xlabel("Number of nodes in the high graph as fraction of $n$")
        plt.ylabel("Fraction covered")
        plt.title(f"Coverage of the nodes in the high graph. Minimum follower out-deg.: {min_out_deg}")
        if results_file:
            with open(results_file, "w") as f:
                json.dump(plot_values, f, sort_keys=True, indent=4, separators=(',', ': '))
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')


if __name__ == "__main__":
    min_out_deg = 10
    res_file = f"../results/high-graph-coverages-min-out-deg-{min_out_deg}.json"
    print(f"Results file: {res_file}")
    HighGraphCoverageAnalyzer.plot_coverages(datasets, results_file=res_file, min_out_deg=min_out_deg,
                                             output_file=f"../figures/high_graph_coverage_min_out-{min_out_deg}.pdf")
