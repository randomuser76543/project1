from globals import datasets

from GraphUtils import GraphUtils as gu

if __name__ == "__main__":
    for file_path, sep, title, directed in datasets[1:2]:
        print("Loading the graph: ", title)
        g, in_degrees = gu.load_graph(file_path, sep, directed)
        print("Retrieving the largest connected component in the 0.01-high graph")
        cc = g.subgraph(
            gu.get_largest_cc_in_high_graph(g=g, in_degrees=in_degrees, num_high_nodes=int(0.01 * g.vcount())))
        print("Computing diameter...")
        print(f"{title} - diameter = {cc.diameter()}")
