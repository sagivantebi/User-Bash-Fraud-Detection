import os
from collections import defaultdict
import aug_functions
import numpy as np
import scipy
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import recall_score, precision_score
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


# Generate random segments for data augmentation
def create_random_segments(command_set, segment_length, total_segments):
    """
    Generates random segments from a set of commands.

    Parameters:
    command_set (set): Set of unique commands.
    segment_length (int): Length of each segment.
    total_segments (int): Number of segments to generate.

    Returns:
    list: List of randomly generated command segments.
    """
    command_list = list(command_set)
    random_command_segments = []

    for _ in range(total_segments):
        random_segment = np.random.choice(command_list, segment_length, replace=True)
        random_command_segments.append(' '.join(random_segment))

    return random_command_segments


def vectorize_commands(command_data, ngram_range=(2, 5), max_features=300):
    """
    Vectorizes text data using CountVectorizer.

    Parameters:
    command_data (list): List of command sequences to be vectorized.
    ngram_range (tuple): The lower and upper boundary of the range of n-values for different n-grams to be extracted. Default is (2, 5).
    max_features (int): Maximum number of features to keep. Default is 300.

    Returns:
    tuple: A tuple containing the vectorized data and the Count vectorizer.
    """
    vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features)
    vectorized_commands = vectorizer.fit_transform(command_data)
    return vectorized_commands, vectorizer


class LSTMSequenceAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
        super(LSTMSequenceAutoencoder, self).__init__()
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.output_linear = nn.Linear(hidden_size, input_size)

    def encode(self, input_sequence):
        _, (hidden_state, _) = self.encoder_lstm(input_sequence)
        return hidden_state.squeeze(0)

    def decode(self, encoded_sequence, sequence_length):
        encoded_sequence = encoded_sequence.unsqueeze(1).repeat(1, sequence_length, 1)
        lstm_output, _ = self.decoder_lstm(encoded_sequence)
        output_sequence = self.output_linear(lstm_output)
        return output_sequence

    def forward(self, input_sequence):
        sequence_length = input_sequence.size(1)
        encoded = self.encode(input_sequence)
        decoded = self.decode(encoded, sequence_length)
        return decoded

def train_lstm_autoencoder(train_data, input_size, hidden_size=128, num_epochs=25, batch_size=32):
    """
    Trains an LSTM autoencoder on the provided training data.

    Parameters:
    train_data (array-like): The training data.
    input_size (int): The number of expected features in the input.
    hidden_size (int): The number of features in the hidden state. Default is 128.
    num_epochs (int): The number of training epochs. Default is 20.
    batch_size (int): The batch size for training. Default is 32.

    Returns:
    model (LSTMSequenceAutoencoder): The trained LSTM autoencoder model.
    mse (numpy array): Mean squared error for the reconstructed training data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMSequenceAutoencoder(input_size, hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # Ensure train_data is a dense array if it comes from TF-IDF
    if isinstance(train_data, scipy.sparse.csr_matrix):
        train_data = train_data.toarray()

    # Reshape train_data to add a feature dimension
    train_tensor = torch.FloatTensor(train_data).unsqueeze(2).to(device)
    train_tensor = train_tensor.permute(0, 2, 1)

    dataset = TensorDataset(train_tensor, train_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train_epoch(model, data_loader, criterion, optimizer, device):
        model.train()
        for batch_data in data_loader:
            inputs, targets = batch_data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    for epoch in range(num_epochs):
        train_epoch(model, data_loader, criterion, optimizer, device)

    model.eval()
    with torch.no_grad():
        reconstructed = model(train_tensor)
        mse = torch.mean((train_tensor - reconstructed) ** 2, dim=2).mean(dim=1).cpu().numpy()

    return model, mse

def combine_anomaly_scores(ae_anomaly_scores, rf_anomaly_probabilities, weight_factors=[0.5, 0.5]):  # Adjusted weights
    return weight_factors[0] * ae_anomaly_scores + weight_factors[1] * rf_anomaly_probabilities



def extract_all_commands(segment_data):
    return set(command for segments in segment_data.values() for segment in segments for command in segment.split())


def split_train_val_data(segment_data, segment_labels, train_ratio=0.75):
    random_split = np.random.permutation(10)
    training_users = [f'User{idx}' for idx in random_split[3:]]
    validation_users = [f'User{idx}' for idx in random_split[:3]]

    return training_users, validation_users


def augment_training_segments(segment_data, segment_labels, training_users, augment_count, benign_data, anomalous_data, command_set):
    for user_id in training_users:
        user_segments = segment_data[user_id][:50]
        user_labels = np.array(segment_labels[user_id][:50])
        augmented_segments, augmented_labels = aug_functions.augment_user_data(
            user_segments, user_labels, augment_count, user_id, benign_data, anomalous_data, command_set
        )
        yield user_id, augmented_segments, augmented_labels


def train_lstm_autoencoder_model(segment_data, text_vectorizer, user_id, lstm_hidden_size, num_epochs):
    vectorized_data = text_vectorizer.transform(segment_data[user_id]).toarray()
    autoencoder_model, reconstruction_errors = train_lstm_autoencoder(
        vectorized_data, vectorized_data.shape[1], lstm_hidden_size, num_epochs
    )
    return autoencoder_model, reconstruction_errors


def compute_test_anomaly_scores(autoencoder_model, test_data, device):
    test_tensor = torch.FloatTensor(test_data.toarray()).unsqueeze(1).to(device)
    with torch.no_grad():
        reconstructed_data = autoencoder_model(test_tensor)
        test_anomaly_scores = torch.mean((test_tensor - reconstructed_data) ** 2, dim=1).mean(dim=1).cpu().numpy()
    return (test_anomaly_scores - min(test_anomaly_scores)) / (max(test_anomaly_scores) - min(test_anomaly_scores))

def find_optimal_threshold(y_true, y_scores):
    """
    Calculate the optimal threshold based on the median and IQR of the anomaly scores.

    Parameters:
    y_true (array-like): True labels of the test data.
    y_scores (array-like): Anomaly scores of the test data.

    Returns:
    float: The calculated optimal threshold.
    """
    # Calculate the median and interquartile range (IQR) of the scores
    median_score = np.median(y_scores)
    iqr = np.percentile(y_scores, 75) - np.percentile(y_scores, 25)

    # Calculate the optimal threshold
    optimal_threshold = median_score + 1.5 * iqr  # Adjust the multiplier based on your needs

    return optimal_threshold

def train_evaluate_anomaly_model(segment_data, segment_labels, training_users, augment_count, benign_data,
                                 anomalous_data, command_set, ae_weight, rf_weight, lstm_hidden_size, num_epochs):
    user_results = defaultdict(list)
    all_labels = []
    all_predictions = []
    user_thresholds = []
    user_vectors = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for user_id, augmented_segments, augmented_labels in augment_training_segments(segment_data, segment_labels,
                                                                                   training_users, augment_count,
                                                                                   benign_data, anomalous_data,
                                                                                   command_set):
        vectorized_train_data, text_vectorizer = vectorize_commands(augmented_segments, ngram_range=(2, 5),
                                                                    max_features=200)
        vectorized_test_data = text_vectorizer.transform(segment_data[user_id][50:])
        autoencoder_model, reconstruction_errors = train_lstm_autoencoder_model(segment_data, text_vectorizer, user_id,
                                                                                lstm_hidden_size, num_epochs)
        test_anomaly_scores = compute_test_anomaly_scores(autoencoder_model, vectorized_test_data, device)

        rf_model = RandomForestClassifier(random_state=25, n_estimators=2000)
        rf_model.fit(vectorized_train_data, augmented_labels.astype(int))
        rf_anomaly_probabilities = rf_model.predict_proba(vectorized_test_data.toarray())[:, 1]

        combined_anomaly_scores = combine_anomaly_scores(test_anomaly_scores, rf_anomaly_probabilities,
                                                         weight_factors=[ae_weight, rf_weight])

        # Ensure no NaN values in segment_labels before passing
        clean_labels = np.array(segment_labels[user_id][50:]).astype(float)
        clean_labels[np.isnan(clean_labels)] = 0
        clean_labels = clean_labels.astype(int)

        # Calculate optimal threshold
        optimal_threshold = find_optimal_threshold(clean_labels, combined_anomaly_scores)
        user_thresholds.append(optimal_threshold)

        # Apply the dynamic threshold directly
        predictions = (combined_anomaly_scores > optimal_threshold).astype(int)

        # Append user_vector (mean, std, max, min of combined_anomaly_scores)
        user_vector = (
        np.mean(combined_anomaly_scores), np.std(combined_anomaly_scores), np.max(combined_anomaly_scores),
        np.min(combined_anomaly_scores))
        user_vectors.append(user_vector)

        selected_anomalies = np.where(predictions == 1)[0][:20]
        user_predictions = np.zeros(len(segment_labels[user_id][50:]))
        if len(selected_anomalies) < 6:
            selected_anomalies = np.argsort(combined_anomaly_scores)[::-1][:6]
        user_predictions[selected_anomalies] = 1

        all_labels.extend(clean_labels)
        all_predictions.extend(user_predictions)

        recall, precision, f1_score = compute_evaluation_metrics(clean_labels, user_predictions)
        user_results['User ID'].append(user_id)
        user_results['Recall'].append(recall)
        user_results['Precision'].append(precision)
        user_results['F1 Score'].append(f1_score)
        print(recall)
        print(precision)
        print(f1_score)
        print(f"Training user: {user_id}")
        print(f"Clean labels distribution: {np.bincount(clean_labels)}")
        print(f"Combined anomaly scores: {combined_anomaly_scores[:10]}")  # Check the first few combined scores

    return user_results, all_labels, all_predictions, user_thresholds, user_vectors



def compute_evaluation_metrics(true_labels, predicted_labels):
    # Ensure both true_labels and predicted_labels contain only binary values
    true_labels = np.array(true_labels).astype(int)
    true_labels = np.clip(true_labels, 0, 1)

    predicted_labels = np.array(predicted_labels).astype(int)
    predicted_labels = np.clip(predicted_labels, 0, 1)

    recall = recall_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return recall, precision, f1_score


def main(is_inference=False, lstm_hidden_size=64, num_epochs=25):
    segment_data, segment_labels = aug_functions.segment_user_data(data_dir, label_file)
    command_set = extract_all_commands(segment_data)
    validation_results = []

    if is_inference:
        training_users, validation_users = split_train_val_data(segment_data, segment_labels)
        print(f'Validation Users: {validation_users}')
    else:
        training_users = [f'User{uid}' for uid in range(10)]
        validation_users = []

    benign_data = {user_id: segments[:50] for user_id, segments in list(segment_data.items())[10:]}
    temp_data = {uid: segment_data[uid] for uid in training_users}
    benign_data.update(temp_data)
    anomalous_data = {uid: [segment_data[uid][i] for i, label in enumerate(segment_labels[uid]) if label == 1] for uid in training_users}

    user_results, all_labels, all_predictions, user_thresholds, user_vectors = train_evaluate_anomaly_model(
        segment_data, segment_labels, training_users, 15, benign_data, anomalous_data, command_set, 0.2, 0.8, lstm_hidden_size, num_epochs
    )

    # Inference for validation users
    for user_id in validation_users:
        validation_results.append(validate_user_anomaly(user_id, segment_data, segment_labels, benign_data, anomalous_data, command_set, 0.2, 0.8, lstm_hidden_size, num_epochs, user_thresholds, user_vectors))

    # Collect predictions for the remaining users (users beyond the first 10)
    inference_predictions = []
    for user_id in segment_data.keys():
        if user_id not in training_users and user_id not in validation_users:
            _, predictions = validate_user_anomaly(user_id, segment_data, segment_labels, benign_data, anomalous_data, command_set, 0.2, 0.8, lstm_hidden_size, num_epochs, user_thresholds, user_vectors)
            inference_predictions.append((user_id, predictions))

    # Save results to CSV
    save_results_to_csv(inference_predictions, 'challengeToFill.csv')

    return validation_results


def validate_user_anomaly(user_id, segment_data, segment_labels, benign_data, anomalous_data, command_set, ae_weight,
                          rf_weight, lstm_hidden_size, num_epochs, user_thresholds, user_vectors):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_segments = segment_data[user_id][:50]
    user_labels = np.array(segment_labels[user_id][:50])
    test_segments = segment_data[user_id][50:]

    # Ensure we handle the case where user_labels might not have exactly 50 entries
    if len(user_labels) < 50:
        user_labels = np.concatenate([user_labels, np.zeros(50 - len(user_labels))])

    augmented_segments, augmented_labels = aug_functions.augment_user_data(user_segments, user_labels, 20, user_id, benign_data,
                                                             anomalous_data, command_set)

    vectorized_train_data, text_vectorizer = vectorize_commands(augmented_segments, ngram_range=(2, 3))
    autoencoder_model, reconstruction_errors = train_lstm_autoencoder_model(segment_data, text_vectorizer, user_id,
                                                                            lstm_hidden_size, num_epochs)
    test_anomaly_scores = compute_test_anomaly_scores(autoencoder_model, text_vectorizer.transform(test_segments),
                                                      device)

    rf_model = RandomForestClassifier(random_state=25, n_estimators=2000)
    rf_model.fit(vectorized_train_data, augmented_labels.astype(int))
    rf_anomaly_probabilities = rf_model.predict_proba(text_vectorizer.transform(test_segments).toarray())[:, 1]

    combined_anomaly_scores = combine_anomaly_scores(test_anomaly_scores, rf_anomaly_probabilities,
                                                     weight_factors=[ae_weight, rf_weight])

    # Ensure no NaN values in segment_labels before passing
    clean_labels = np.array(segment_labels[user_id][50:])
    if clean_labels.size == 0:
        clean_labels = np.zeros(len(test_segments))
    else:
        clean_labels = clean_labels.astype(float)
        clean_labels[np.isnan(clean_labels)] = 0
        clean_labels = clean_labels.astype(int)

    # Calculate the threshold based on the median and IQR of combined anomaly scores
    median_score = np.median(combined_anomaly_scores)
    iqr = np.percentile(combined_anomaly_scores, 75) - np.percentile(combined_anomaly_scores, 25)
    optimal_threshold = median_score + 1.5 * iqr  # Adjust the multiplier based on your needs

    # Apply the dynamic threshold directly
    predictions = (combined_anomaly_scores > optimal_threshold).astype(int)

    selected_anomalies = np.where(predictions == 1)[0]
    if len(selected_anomalies) < 6:
        selected_anomalies = np.argsort(combined_anomaly_scores)[::-1][:6]
    if len(selected_anomalies) > 10:
        selected_anomalies = selected_anomalies[:10]

    selected_anomalies = selected_anomalies.astype(int)  # Ensure selected_anomalies contains integer values

    user_predictions = np.zeros(len(test_segments))
    user_predictions[selected_anomalies] = 1

    # Update clean_labels with selected anomalies, ensuring a maximum of 10 anomalies
    clean_labels.fill(0)  # Reset all to benign
    for idx in selected_anomalies:
        clean_labels[idx] = 1

    # Ensure the clean labels distribution is [90,10] or [88,12]
    if np.sum(clean_labels) > 10:
        anomalies_count = 10
        benign_count = len(clean_labels) - anomalies_count
    else:
        anomalies_count = np.sum(clean_labels)
        benign_count = len(clean_labels) - anomalies_count

    clean_labels[:benign_count] = 0
    clean_labels[benign_count:benign_count + anomalies_count] = 1

    return clean_labels, user_predictions.astype(int)




def save_results_to_csv(predictions, file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    modified_lines = [lines[0]]  # Copy the header

    for i, line in enumerate(lines[1:]):  # Skip the header
        if i < 10:
            # Keep the first 10 users unchanged
            modified_lines.append(line)
            continue
        else:
            temp = line.split(',')[:51]  # Include the user ID and first 50 values (benign segments)

            # Ensure predictions have the correct size
            user_idx = i - 10
            if user_idx < len(predictions):
                pred_values = predictions[user_idx][1].tolist()  # Get the predicted values

                # First 50 chunks are benign (0)
                pred_values = [0] * 50 + pred_values[50:]

                if len(pred_values) < 100:
                    pred_values.extend([0] * (100 - len(pred_values)))  # Extend with zeros if needed

                temp.extend(pred_values)
            else:
                temp.extend([0] * 100)  # Extend with zeros if no predictions are available

            modified_lines.append(','.join(map(str, temp)))

    # Save to a new file if file already exists
    vers = 0
    filename = f'challengeToFill_{vers}.csv'
    while os.path.exists(filename):
        vers += 1
        filename = f'challengeToFill_{vers}.csv'
    print(f'Writing to {filename}')
    with open(filename, 'w') as file:
        for line in modified_lines:
            file.write(f'{line}\n')
    print('File written successfully')


if __name__ == "__main__":
    # Define the directory and label paths
    data_dir = 'FraudedRawData'
    label_file = 'challengeToFill.csv'

    # Run the main function
    main(is_inference=True, lstm_hidden_size=64, num_epochs=30)
