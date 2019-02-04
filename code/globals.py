from collections import namedtuple
Dataset = namedtuple("Dataset", ["path", "sep", "title", "directed"])

datasets = [Dataset("/User/data/networks/soc-epinions/soc-Epinions1.txt", "\t", "Epinions", True),
            Dataset("/User/data/networks/soc-slashdot/soc-Slashdot0902.txt", "\t", "Slashdot", True),
            Dataset("/User/data/networks/com-youtube.ungraph.csv", "\t", "Youtube", False),
            Dataset("/User/data/networks/com-orkut.ungraph.txt", "\t", "Orkut", False),
            Dataset("/User/data/networks/soc-twitter-higgs/higgs-social_network.edgelist", " ",
                    "Twitter-Higgs", True),
            Dataset("/User/data/networks/soc-livejournal/soc-LiveJournal1.txt", "\t", "LiveJournal", True),
            Dataset("/User/data/networks/soc-pokec/soc-pokec-relationships.txt", "\t", "Pokec", True),
            Dataset("/User/data/networks/wiki-topcats.txt", " ", "Wiki-topcats", True),
            Dataset("/User/data/networks/com-dblp.ungraph.txt", "\t", "DBLP", False),
            Dataset("/file/not/included/here/due/to/size", " ", "Twitter-MPI-SWS", True),
            Dataset("/file/not/included/here/due/to/size", " ", "Tumblr", True)
            ]
