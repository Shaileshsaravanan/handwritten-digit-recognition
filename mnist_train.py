import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K

# Ensure matplotlib plots are displayed inline
%matplotlib inline

# Set backend image data format
K.set_image_data_format('channels_first')

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Load and preprocess data
def load_and_preprocess_data(train_csv_path):
    # Load data from CSV
    data = pd.read_csv(train_csv_path).values
    
    # Extract labels (first column)
    labels = data[:, 0]
    data = data[:, 1:]
    
    # Split data into training and validation sets
    train_data = data[:35000, :]
    valid_data = data[35000:, :]
    train_labels = labels[:35000]
    valid_labels = labels[35000:]
    
    # Reshape and normalize data
    train_data = train_data.reshape(-1, 1, 28, 28).astype('float32') / 255
    valid_data = valid_data.reshape(-1, 1, 28, 28).astype('float32') / 255
    
    # Convert labels to one-hot encoding
    train_labels = np_utils.to_categorical(train_labels)
    valid_labels = np_utils.to_categorical(valid_labels)
    
    return train_data, train_labels, valid_data, valid_labels

# Plot label distribution
def plot_label_distribution(labels):
    sns.countplot(labels)
    plt.title("Label Distribution")
    plt.show()

# Build the CNN model
def create_model(input_shape=(1, 28, 28), num_classes=10):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Evaluate and save the model
def evaluate_and_save_model(model, valid_data, valid_labels, model_path="model.h5", model_json_path="model.json"):
    scores = model.evaluate(valid_data, valid_labels, verbose=0)
    print(f"CNN Accuracy: {scores[1] * 100:.2f}%")
    
    # Save model weights
    model.save(model_path)
    print(f"Model weights saved as {model_path}")
    
    # Save model architecture to JSON
    model_json = model.to_json()
    with open(model_json_path, "w") as json_file:
        json_file.write(model_json)
    print(f"Model architecture saved as {model_json_path}")

def main():
    # Paths to the data
    train_csv_path = "../input/train.csv"
    
    # Load and preprocess the data
    train_data, train_labels, valid_data, valid_labels = load_and_preprocess_data(train_csv_path)
    
    # Plot label distribution
    plot_label_distribution(pd.read_csv(train_csv_path)['label'])
    
    # Create and train the model
    model = create_model()
    model.fit(train_data, train_labels, validation_data=(valid_data, valid_labels), epochs=10, batch_size=200, verbose=2)
    
    # Evaluate and save the model
    evaluate_and_save_model(model, valid_data, valid_labels)

if __name__ == "__main__":
    main()