import pickle

def load_embeddings_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        embeddings_with_labels = pickle.load(f)
    return embeddings_with_labels

# Example usage
if __name__ == '__main__':
    embeddings_file_path = './data/embeddings_with_labels.pkl'
    embeddings_with_labels = load_embeddings_from_pickle(embeddings_file_path)

    # Print the loaded embeddings and their corresponding labels
    for label, embedding in embeddings_with_labels.items():
        print(f"Label: {label}, Embedding shape: {embedding.shape if hasattr(embedding, 'shape') else len(embedding)}")
