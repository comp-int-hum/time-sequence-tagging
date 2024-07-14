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
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import gzip
from collections import defaultdict, Counter

QUOTE_PUNCTS = [
    # ASCII
    '"', "'", "``", "’",     

    # Smart quotes
    "“", "”", "‘", "’",     

    # Angle
    "«", "»", "‹", "›",     

    # Brackets
    "「", "」", "『", "』",    

    # Other
    "„", "”", "‚"      
]

def is_dialogue(sentence):
    cleaned_sent = sentence.strip()
    if cleaned_sent[0] in QUOTE_PUNCTS or cleaned_sent[-1] in QUOTE_PUNCTS:
        return False
    return True



# def ids_to_line_nums_dict(pg_path):
#     reverse_dict = {}
#     with jsonlines.open(pg_path, "r") as reader:
#         for i, text in enumerate(reader):
#             reverse_dict[text["id"]] = i
#     return reverse_dict

# def convert_ids_to_line_nums(reverse_dict, texts):
#     line_nums = []
#     for text in texts:
#         line_nums.append(reverse_dict[text["id"]])
#     return line_nums

# def get_texts(pg_path, line_nums):
#     texts = {}
#     with jsonlines.open(pg_path, "r") as reader:
#         for i, text in enumerate(reader):
#             if i in line_nums:
#                 texts[text["id"]] = text
#     return texts

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object {type(obj)} cannot be serialized")


# Assume file has ids in sorted order
def append_metadata_to_stats(pg_path, text_statistics):
    with jsonlines.open(pg_path, "r") as reader:
        for text in reader:
            if text["id"] in text_statistics:
                text_statistics[text["id"]] += (text["title"], text["author"])

def gather_text_stats(chapters):
    ch_lens = []
    par_lens = []
    ch_dialogue = []

    # Loop over chapters
    for (_, ch) in chapters:
        _, original_text = ch
        
        # Define ch_len and dialogue
        ch_len = 0
        dialogue = 0
        
         # Loop over paragraphs
        for pars in original_text:
            par_lens.append(len(pars))
            ch_len += len(pars)
            for sent in pars:
                if is_dialogue(sent):
                    dialogue += 1
        
        ch_lens.append(ch_len)
        ch_dialogue.append(dialogue / ch_len if ch_len else 0)

    return (np.average(ch_lens), np.std(ch_lens), np.average(par_lens), np.std(par_lens), np.average(ch_dialogue), np.std(ch_dialogue))

def kmeans_cluster_texts(text_tuples, image_path, num_clusters):
    # Set up features and labels
    text_data = np.array(text_tuples)
    features = text_data[:, 0:-2].astype(float)
    labels = text_data[:, -2:]
    # print(f"Features: {features.shape}")
    
    # Normalize
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)
    
    # K-means cluster
    kmeans = KMeans(n_clusters = num_clusters)
    kmeans.fit(standardized_features)
    
    # PCA for visualization
    pca = PCA(n_components = 2)
    reduced_features = pca.fit_transform(standardized_features)
    
    # Scatter for individual texts
    plt.figure(figsize=(10, 8))
    scatter_plot = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c = kmeans.labels_, cmap = "viridis")
    
    # Scatter for cluster centers
    centers = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1], c = "red", s = 150, alpha = 1.0, marker = "X")
    
    # Add legend, title
    cluster_legend = plt.legend(*scatter_plot.legend_elements(), title="Clusters")
    plt.gca().add_artist(cluster_legend)
    plt.title("K-means Clustering")
    
    # Annotate certain number of indices
    sample_indices = random.sample(range(len(labels)), 10)
    for i, txt in enumerate(labels):
        if i in sample_indices:
            plt.annotate(txt, (reduced_features[i, 0], reduced_features[i, 1]), fontsize=6, alpha=1.0)
            
	# Save and show
    plt.savefig(image_path)
    plt.show()
    
    return list(kmeans.labels_), list(kmeans.cluster_centers_)
    
def gmm_cluster(text_tuples, image_path, num_clusters):
    text_data = np.array(text_tuples)
    features = text_data[:, 0:-2].astype(float)
    labels = text_data[:, -2:]
    # print(f"Features: {features.shape}")
    
    # Normalize
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)
    
    # Gaussian Mixture Model cluster
    gmm = GaussianMixture(n_components=num_clusters)
    gmm.fit(standardized_features)
    cluster_labels = gmm.predict(standardized_features)
    cluster_centers = gmm.means_
    
    # PCA for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(standardized_features)
    
    # Scatter for individual texts
    plt.figure(figsize=(10, 8))
    scatter_plot = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap="viridis")
    
    # Scatter for cluster centers
    centers = pca.transform(cluster_centers)
    plt.scatter(centers[:, 0], centers[:, 1], c="red", s=150, alpha=1.0, marker="X")
    
    # Add legend, title
    cluster_legend = plt.legend(*scatter_plot.legend_elements(), title="Clusters")
    plt.gca().add_artist(cluster_legend)
    plt.title("Gaussian Mixture Model Clustering")
    
    # Annotate certain number of indices
    sample_indices = random.sample(range(len(labels)), 10)
    for i, txt in enumerate(labels):
        if i in sample_indices:
            plt.annotate(txt, (reduced_features[i, 0], reduced_features[i, 1]), fontsize=6, alpha=1.0)
            
    # Save and show
    plt.savefig(image_path)
    plt.show()
    
    return list(cluster_labels), list(cluster_centers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", dest="collection", help = "Encoded collection")
    parser.add_argument("--pg_path", dest="pg_path", help = "Path for scraped Project Gutenberg file")
    parser.add_argument("--compressed_output", dest="compressed_output")
    parser.add_argument("--sample_output", dest = "sample_output")
    parser.add_argument("--image", dest="image", help = "output path for image")
    parser.add_argument("--clusters", dest = "clusters", default = 5, type = int)
    args, rest = parser.parse_known_args()
    
    # Read and cluster
    with open_file(args.collection, "r") as file:
        with jsonlines.Reader(file) as collection:
            text_statistics = {}
            for idx, doc in tqdm(enumerate(collection), desc="Collecting Metadata"):
                text_stats = gather_text_stats(list(doc["encoded_segments"].items()))
                if 0 not in text_stats[0:4]:
                    text_statistics[doc["id"]] = text_stats
            
            append_metadata_to_stats(args.pg_path, text_statistics)
            cluster_labels, cluster_centers = gmm_cluster(list(text_statistics.values()), args.image, args.clusters)

    # Add metadata and cluster labels and rewrite out
    with open_file(args.collection, "r") as file:
        with jsonlines.Reader(file) as collection:
            group_by_cluster = {
                "metadata": {
                    "cluster counts": sorted(list(Counter(cluster_labels).items())),
                    "stat_labels": [
                        "Avg chapter length (in par)",
                        "StDev for chapter len",
                        "Avg paragraph len (in sentences)",
                        "StDev for par len",
                        "Avg ratio of dialogue in chapter",
                        "StDev for ratio of dialogue in chapter"
                    ],
                    "cluster_centers": cluster_centers,
                    "cluster_members": {str(i): [] for i in range(args.clusters)}
                },
                **{str(i): [] for i in range(args.clusters)}
            }
            
            with gzip.open(args.compressed_output, mode = "wt") as output:
                for c_label, text_stats, doc in tqdm(zip(cluster_labels, text_statistics.values(), collection), desc="Writing out"):
                    c_label_str = str(c_label)
                    doc.update({
                        "cluster": c_label_str,
                        "title": text_stats[-2],
                        "author": text_stats[-1],
                        "stat_labels": ["Avg chapter length (in par)", "StDev for chapter len", "Avg paragraph len (in sentences)", "StDev for par len"],
                        "stats": text_stats[:-2]
                    })
                    output.write(json.dumps(doc) + "\n")
                    del doc["encoded_segments"]
                    
                    # Optionally: switch to reservoir sampling
                    if len(group_by_cluster[c_label_str]) < 10:
                        group_by_cluster[c_label_str].append(doc)
                    group_by_cluster["metadata"]["cluster_members"][c_label_str].append(doc["title"] + " by " + doc["author"])
            with open(args.sample_output, "w", encoding = "utf-8") as sample_output:
                json.dump(group_by_cluster, sample_output, default = convert_numpy)
            

            






        
    
    
    


                    
    
                
