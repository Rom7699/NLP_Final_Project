import json
from collections import Counter
from nltk.corpus import stopwords
from compare_clustering_solutions import evaluate_clustering
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

similarity_threshold = 0.6
max_iterations = 15
length_weight=0.6



###### DATA PROCESSING ######

def normalize_text(text):
    """
    Normalizes text by lowercasing and remove spaces.
    """
    return text.lower().strip().replace("\r\n", "\n")  # Convert `\r\n` to `\n`

def load_data(data_file):
    """
    Loads the requests data from a CSV file and normalizes.
    """
    requests = pd.read_csv(data_file)
    requests["text"] = requests["text"].apply(normalize_text)
    return requests

def preprocess_text(text):
    """
    Tokenizes and lemmatizes text.
    """
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text.lower())  # Tokenize and lowercase
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized_tokens)


def get_embeddings_vectors(requests):
    """
    Computes embeddings for the given requests using a SentenceTransformer model.

    :param requests: DataFrame containing the requests (with a "text" column).
    :return: A NumPy array of embedding vectors corresponding to each request.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(requests["text"])
    return embeddings


###### CLUSTERING ######

def find_best_cluster_for_request(request_embedding, clusters):
    """
    Finds the best matching cluster using the dot product.

    :param request_embedding: The embedding vector of the request.
    :param clusters: List of cluster dictionaries.
    :return: Tuple (best_cluster_index, best_similarity) where best_cluster_index is
             the index of the best matching cluster
             and best_similarity is the dot product similarity value.
    """
    if not clusters:
        return None, None  # No clusters exist

    cluster_centroids = np.array([c["centroid"] for c in clusters])

    request_embedding = np.array(request_embedding)
    cluster_centroids = np.array(cluster_centroids)

    # Compute dot product
    similarities = np.dot(request_embedding, cluster_centroids.T)

    # Find the best cluster
    best_cluster_index = np.argmax(similarities)
    best_similarity = similarities[best_cluster_index]

    return best_cluster_index, best_similarity


def update_cluster_centroid(cluster, embeddings):
    """
    Updates the centroid of a cluster based on the embeddings of its requests.

    :param cluster: A dictionary with at least the key "indices" (list of request indices).
    :param embeddings: A NumPy array of all embeddings.
    """
    indices = cluster["indices"]
    if indices:
        cluster["centroid"] = np.mean(embeddings[indices], axis=0)

def cluster_requests(requests, embeddings, min_cluster_size):
    """
       Clusters the given requests into groups.
       this function processes all requests in each iteration, assigning each to its best cluster (or creating a new
       cluster if none meet the threshold) and updating centroids.

       :param requests: List of user requests (only used for indexing).
       :param embeddings: NumPy array of embedding vectors for the requests.
       :param similarity_threshold: Minimum cosine similarity to join an existing cluster.
       :param min_cluster_size: Minimum number of requests for a cluster to be valid.
       :param max_iterations: Maximum number of iterations to run the assignment loop.
       :return: A list of clusters. Each cluster is a dictionary with keys "indices" and "centroid".
       """
    clusters = []
    request_to_cluster = [-1] * len(requests)  # Track which cluster each request belongs to.
    for iteration in range(max_iterations):
        print(f"Iteration: {iteration+1}")
        changed = False

        for i, emb in enumerate(embeddings):
            # Find the best cluster for the current request.
            best_cluster_index, best_similarity = find_best_cluster_for_request(emb, clusters)

            # Check if the best match meets the threshold.
            if best_similarity is not None and best_similarity >= similarity_threshold:
                if request_to_cluster[i] != best_cluster_index:
                    # if it was assigned, remove it from that cluster
                    if request_to_cluster[i] != -1:
                        prev_cluster = clusters[request_to_cluster[i]]
                        if i in prev_cluster["indices"]:
                            prev_cluster["indices"].remove(i)
                        update_cluster_centroid(prev_cluster, embeddings)

                    # Add to the best matching cluster.
                    clusters[best_cluster_index]["indices"].append(i)
                    request_to_cluster[i] = best_cluster_index
                    update_cluster_centroid(clusters[best_cluster_index], embeddings)
                    changed = True
            else:
                # No cluster is similar enough. create a new cluster with this request
                new_cluster = {"indices": [i], "centroid": emb}
                clusters.append(new_cluster)
                request_to_cluster[i] = len(clusters) - 1
                changed = True

        # If no request changed its cluster assignment we have converged
        if not changed:
            print(f"Converged after {iteration + 1} iterations.")
            break

    # Separate clusters into valid ones and unclustered requests.
    valid_clusters = []
    valid_indices = set()
    for cluster in clusters:
        if len(cluster["indices"]) >= min_cluster_size:
            valid_clusters.append(cluster)
            valid_indices.update(cluster["indices"])

    # All indices (requests) that are not in valid clusters are unclustered.
    all_indices = set(range(len(requests)))
    unclustered = list(all_indices - valid_indices)

    return valid_clusters, unclustered


###### LABELS ######

def get_top_k_significant_words(texts, top_k):
    """
    Identifies the top K most significant words in a cluster based on frequency,
    after preprocessing and filtering for content words (nouns and verbs).

    :param texts: List of texts in the cluster.
    :param top_k: Number of top words to select.
    :return: List of the top significant words (nouns or verbs).
    """
    preprocessed_texts = [preprocess_text(text) for text in texts]
    stop_words = set(stopwords.words("english"))

    vectorizer = CountVectorizer(stop_words=None)
    X = vectorizer.fit_transform(preprocessed_texts)

    if X.shape[1] == 0:
        return []

    # Sum counts for each word.
    counts = np.asarray(X.sum(axis=0)).flatten()
    feature_names = vectorizer.get_feature_names_out()

    word_counts = list(zip(feature_names, counts))

    # Filter to only keep words that are nouns or verbs and not in stop words.
    filtered_words = []
    for word, count in word_counts:
        tag = nltk.pos_tag([word])[0][1]
        if (tag.startswith("NN") or tag.startswith("VB")) and (word.lower() not in stop_words):
            filtered_words.append((word.lower(), count))

    filtered_words.sort(key=lambda x: x[1], reverse=True)

    top_words = [word for word, count in filtered_words[:top_k]]
    return top_words



def pos_pattern_bonus(ngram, top_significant_words):
    """
    Computes a score based on the POS pattern of the candidate n-gram,
    and adds extra score if the candidate contains one or more of the top significant words.
      - If the phrase has at least 3 tokens and follows an imperative pattern (verb, then determiner/possessive, then noun)
      - If the phrase is a noun phrase (all tokens are DT, JJ, or NN types and ends with a noun)
      - If the phrase is exactly two tokens and is a noun noun or verb noun
      - Additionally, for every occurrence of a top significant word in the phrase, multiply the score by 4
    """
    tokens = nltk.word_tokenize(ngram)
    tags = [tag for word, tag in nltk.pos_tag(tokens)]
    score = 1.0

    INVALID_START_TAGS = INVALID_END_TAGS = {"DT", "PRP$", "TO", "IN", "CC"}

    if tags[-1] in INVALID_END_TAGS:
        return 0  # we ensure this phrase is never selected

    if tags[0] in INVALID_START_TAGS:
        return 0

         # Imperative phrase:
    if len(tags) >= 3:
        if tags[0] in {"VB", "VBP", "VBZ", "VBG", "VBN"} and tags[1] in {"DT", "PRP$"} and tags[2] in {"NN", "NNS",
                                                                                                       "NNP",
                                                                                                       "NNPS"}:
            score *= 1.2

        # Noun phrase:
    if len(tags) >= 2:
        noun_phrase_tags = {"DT", "JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS"}
        if all(tag in noun_phrase_tags for tag in tags) and tags[-1] in {"NN", "NNS", "NNP", "NNPS"}:
            score *= 1.2

    if len(tags) == 2:
        # Noun–Noun pattern.
        if tags[0] in {"NN", "NNS", "NNP", "NNPS"} and tags[1] in {"NN", "NNS", "NNP", "NNPS"}:
            score *= 1.2
        # Verb–Noun pattern.
        if tags[0] in {"VB", "VBP", "VBZ", "VBG", "VBN"} and tags[1] in {"NN", "NNS", "NNP", "NNPS"}:
            score *= 1.2

    # Extra score for top significant words.
    for word in top_significant_words:
        if word in tokens:
            score *= 4

    return score

def generate_cluster_label_from_texts(texts):
    """
    Generates a label for a cluster using CountVectorizer to extract candidate n-grams,
    then applies a POS check and computes a score that includes:
      - frequency,
      - length penalty (number_of_words ** length_weight)
      - significant words bonus
      - POS pattern bonus
    """

    vectorizer = CountVectorizer(ngram_range=(2, 5))
    X = vectorizer.fit_transform(texts)

    counts = X.sum(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()

    # compute top 3 significant words
    top_significant_words = get_top_k_significant_words(texts, top_k=3)

    # compute the score of each candidate
    candidates = []
    for ngram, count in zip(feature_names, counts):
        score = (count * pos_pattern_bonus(ngram, top_significant_words)) * (len(ngram.split()) ** length_weight)
        candidates.append((ngram, score))

    best_ngram, _ = max(candidates, key=lambda x: x[1])
    return best_ngram.capitalize()

def assign_cluster_labels(clusters, requests):
    """
    Assigns labels to clusters by applying n-gram on the texts in each cluster.

    :param clusters: List of cluster dictionaries.
    :param requests: DataFrame containing the original requests.
    :return: List of labels (one per cluster).
    """
    labels = []
    for cluster in clusters:
        texts = [requests.iloc[i]["text"] for i in cluster["indices"]]
        label = generate_cluster_label_from_texts(texts)
        labels.append(label)
    return labels



###### REPRESENTATIVES ######

def get_representatives_for_clusters(clusters, requests, embeddings, num_representatives):
    """
     For each valid cluster, selects a set of representative request indices using greedy algorithm.

     For each cluster:
        - It extracts the local embeddings from the global embeddings based on the cluster indices.
        - It computes the cluster centroid.
        - It calls get_cluster_representatives to select a set of local representative indices.
        - Then, it maps these local indices back to the global indices.

    :param clusters: List of cluster dictionaries, each containing:
                        - "indices": list of global indices of requests in the cluster.
                        - "centroid": the cluster centroid (a 1D numpy array).
    :param embeddings: NumPy array of embeddings for all requests.
    :param num_representatives: Number of representative requests to select per cluster.
    :return: A list of lists, where each inner list contains the global indices of the representatives for that cluster.
    """
    clusters_representatives = []
    for cluster in clusters:
        indices = cluster["indices"]
        cluster_embeddings = embeddings[indices]
        cluster_texts = requests.iloc[cluster["indices"]]["text"].tolist()
        centroid = cluster["centroid"]
        representatives = get_cluster_representatives(cluster_embeddings, centroid, cluster_texts ,num_representatives)
        clusters_representatives.append([indices[rep] for rep in representatives])
    return clusters_representatives


def get_cluster_representatives(embeddings, centroid, texts, num_representatives):
    """
    Selects representative indices from a single cluster using scoring system.

    :param embeddings: NumPy array of embeddings for requests in the cluster.
    :param centroid: The centroid of the cluster.
    :param texts: List of request texts in the cluster
    :param num_representatives: Number of representatives
    :return: List of selected representative indices (relative to the cluster).
    """

    # find the most frequent request
    request_counts = Counter(texts)
    most_frequent_request = max(request_counts, key=request_counts.get)  # Most common request
    first_rep = texts.index(most_frequent_request)  # Index of first representative
    representatives = [first_rep]

    # Compute distance to centroid
    distances_to_centroid = np.linalg.norm(embeddings - centroid, axis=1)

    # Score and select the remaining representatives
    while len(representatives) < num_representatives:
        candidates = [i for i in range(len(embeddings)) if i not in representatives]
        best_candidate = None
        max_score = -np.inf

        for candidate in candidates:
            candidate_embedding = embeddings[candidate]

            diversity = np.mean([np.linalg.norm(candidate_embedding - embeddings[r]) for r in representatives])

            # Frequency: Normalize instance count (higher is better)
            instance_count = request_counts[texts[candidate]] / max(request_counts.values())

            # Relevance: Negative distance to centroid (closer is better)
            relevance = -distances_to_centroid[candidate]

            score = (3.0 * diversity) + (1.5 * instance_count) + (0.5 * relevance)

            if score > max_score:
                max_score = score
                best_candidate = candidate

        representatives.append(best_candidate)

    return representatives



def build_clustering_output(valid_clusters, cluster_labels, unclustered, requests, clusters_representatives_list):
    """
        Builds the final clustering output as a dictionary with clustered and unclustered requests.
    """
    cluster_list = []
    for cluster, label, representatives in zip(valid_clusters, cluster_labels, clusters_representatives_list):
        cluster_texts = requests.iloc[cluster["indices"]]["text"].tolist()
        cluster_representatives_text = requests.iloc[representatives]["text"].tolist()
        cluster_list.append({
            "cluster_name": label,
            "requests": cluster_texts,
            "representatives": cluster_representatives_text,
        })
    unclustered_texts = requests.iloc[unclustered]["text"].tolist() if unclustered else []
    return {"cluster_list": cluster_list, "unclustered": unclustered_texts}


def analyze_unrecognized_requests(data_file, output_file, num_representatives, min_cluster_size):
   """
   Main function that loads data, computes embeddings, clusters the requests,
    selects representative requests for each cluster, assigns labels to clusters, and writes
    the final clustering output to a JSON file
   """
   requests = load_data(data_file)
   embeddings = get_embeddings_vectors(requests)
   valid_clusters_list, unclustered_list = cluster_requests(requests, embeddings, int(min_cluster_size))

   clusters_representatives_list = get_representatives_for_clusters(valid_clusters_list, requests, embeddings, int(num_representatives))

   cluster_labels = assign_cluster_labels(valid_clusters_list, requests)

   clustering_output = build_clustering_output(valid_clusters_list, cluster_labels, unclustered_list, requests, clusters_representatives_list)

   with open(output_file, "w") as fin:
       json.dump(clustering_output, fin, indent=4)



if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['num_of_representatives'],
                                  config['min_cluster_size'])

    # todo: evaluate your clustering solution against the provided one
    #evaluate_clustering(config['example_solution_file'], config['example_solution_file'])  # invocation example
    evaluate_clustering(config['example_solution_file'], config['output_file'])


