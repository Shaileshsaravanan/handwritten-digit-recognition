import numpy as np
from network import Network
import mnist

# Load data
train_images = mnist.train_images()  # [60000, 28, 28]
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Constants
NUM_CLASSES = 10
INPUT_SHAPE = 784  # 28x28 flattened
BATCH_SIZE = 32  # Use mini-batch gradient descent for faster training
EPOCHS = 10  # Increase epochs for better training
LEARNING_RATE = 0.001
WEIGHTS_FILE = 'weights.pkl'

# Data processing
x_train = train_images.reshape(-1, INPUT_SHAPE).astype('float32') / 255.0  # Flatten and normalize
y_train = np.eye(NUM_CLASSES)[train_labels]  # Convert labels to one-hot encoding

x_test = test_images.reshape(-1, INPUT_SHAPE).astype('float32') / 255.0  # Flatten and normalize
y_test = np.eye(NUM_CLASSES)[test_labels]  # Convert labels to one-hot encoding

# Initialize and configure the network
net = Network(
    num_nodes_in_layers=[INPUT_SHAPE, 20, NUM_CLASSES], 
    batch_size=BATCH_SIZE,
    num_epochs=EPOCHS,
    learning_rate=LEARNING_RATE, 
    weights_file=WEIGHTS_FILE
)

# Train the network
print("Training...")
net.train(x_train, y_train)

# Test the network
print("Testing...")
accuracy = net.test(x_test, y_test)

print(f"Test Accuracy: {accuracy * 100:.2f}%")