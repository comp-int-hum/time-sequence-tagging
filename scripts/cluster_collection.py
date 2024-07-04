import argparse
import json
import jsonlines
import random
from utility import open_file
import gzip
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA

def ids_to_line_nums_dict(pg_path):
    reverse_dict = {}
    with jsonlines.open(pg_path, "r") as reader:
        for i, text in enumerate(reader):
            reverse_dict[text["id"]] = i
    return reverse_dict

def convert_ids_to_line_nums(reverse_dict, texts):
    line_nums = []
    for text in texts:
        line_nums.append(reverse_dict[text["id"]])
    return line_nums

def get_texts(pg_path, line_nums):
    texts = {}
    with jsonlines.open(pg_path, "r") as reader:
        for i, text in enumerate(reader):
            if i in line_nums:
                texts[text["id"]] = text
    return texts

# Assume file has ids in sorted order
def append_years(pg_path, text_statistics):
    with jsonlines.open(pg_path, "r") as reader:
        for text in reader:
            if text["id"] in text_statistics:
                text_statistics[text["id"]] += (text["year"], text["title"], text["author"])

def gather_text_stats(chapters):
    ch_lens = []
    par_lens = []

    # Loop over chapters
    print(type(chapters))
    for (_, ch) in chapters:
        _, original_text = ch
        ch_lens.append(len(original_text))

        # Loop over paragraphs
        for pars in original_text:
            par_lens.append(len(pars))

    return (np.average(ch_lens), np.std(ch_lens), np.average(par_lens), np.std(par_lens))


def plot_years():
    pass

def plot_avg_length():
    pass

def plot_avg_paragraphs_per_chapter():
    pass

def plot_avg_sentences_per_paragraph():
    pass

def cluster_texts(text_tuples, num_clusters):
    data = np.array(text_tuples)
    
    features = data[:, 1:-2].astype(float)
    labels = data[:, -2:]

    kmeans = KMeans(n_clusters = num_clusters)
    kmeans.fit(features)

    print(f"Cluster centers: ", kmeans.cluster_centers_)
    print(f"Labels: ", kmeans.labels_)

    # print(f"Original Labels:\n", labels)

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter_plot = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=kmeans.labels_, cmap='viridis')

    centers = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=0.6, marker='X')

    # Adding legend
    legend1 = plt.legend(*scatter_plot.legend_elements(), title="Clusters")
    plt.gca().add_artist(legend1)

    plt.title("K-means Clustering")
    # plt.xlabel("")
    # plt.ylabel("")
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", dest="collection", help = "Encoded collection")
    parser.add_argument("--pg_path", dest="pg_path", help = "Path for scraped Project Gutenberg file")
    parser.add_argument("--output", dest="output")
    parser.add_argument("--clusters", dest = "clusters", default = 8, type = int)
    args, rest = parser.parse_known_args()

    collection = jsonlines.Reader(open_file(args.collection, "r"))
    text_statistics = {}
    # writer = jsonlines.Writer(gzip.open(args.output, mode="wt"))

    for idx, doc in tqdm(enumerate(collection), desc="Collecting Metadata"):
        # print(f"Doc keys: {doc['original_text'].keys()}")
        print(doc.keys())
        text_statistics[doc["id"]] = gather_text_stats(list(doc["encoded_segments"].items()))
    
    append_years(args.pg_path, text_statistics)
    # combined_tuples = [(key,) + value for key, value in text_statistics.items()]
    # cluster_texts(combined_tuples)
    cluster_texts(text_statistics.values(), args.clusters)
    






        
    
    
    


                    
    
                
