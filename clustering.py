from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import argparse
import csv

def embedding_clustering(embedding_file):
    # generate embeddings
    model = SentenceTransformer('all-MiniLM-L12-v2') # Example model
    with open(embedding_file, 'r') as f:
        sentences = [line.strip() for line in f.readlines()]

    embeddings = model.encode(sentences)
    return embeddings, sentences

# to find optimal number of clusters
def compute_wcss(embeddings):
    wcss = []
    for i in range(1, 11):  
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(embeddings)
        wcss.append(kmeans.inertia_)
        print(f"WCSS for {i} clusters: {kmeans.inertia_}")  # WCSS for each k
    return wcss

def plot_elbow(wcss):
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Plot for K Means')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig('elbow.png')
    plt.show()

def clustering(embeddings, sentences):
    similarity_matrix = cosine_similarity(embeddings)

    num_clusters = 4 # Example number of clusters
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_

    # Mapping sentences to their clusters
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences.setdefault(cluster_id, []).append(sentences[sentence_id])
    return clustered_sentences

'''
def get_mean_scores(clustered_sentences, responses_file):
    mean_scores = []

    # Open the file and create a csv reader object
    with open(responses_file, newline='') as responses:
        for cluster in clustered_sentences:
            scores = []
            for sentence in clustered_sentences[cluster]:
                reader = csv.DictReader(responses)
                for row in reader:
                    if row['prompts'] == sentence:
                        score = row['scores']
                        scores.append(float(score))
                        break
            mean_scores.append(np.mean(scores))

    return mean_scores
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', type=str, default='./data/red_teams_small.txt')
    #parser.add_argument('--responses', type=str, default='./responses/babbage002-harmless-single.txt')
    args = parser.parse_args()
    return(args)

if __name__ == '__main__':
    args = parse_args()
    embeddings, sentences = embedding_clustering(args.embedding)
    wcss = compute_wcss(embeddings)
    print(wcss)
    plot_elbow(wcss)
    clustered_sentences = clustering(embeddings, sentences)
    print(clustered_sentences)
    #mean_scores = get_mean_scores(clustered_sentences, args.responses)
    #print(mean_scores)

