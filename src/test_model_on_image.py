import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load model
model = load_model("asl_model.h5")
labels = [chr(i) for i in range(65, 91)]

# Load test data
df = pd.read_csv("data/sign_mnist_test.csv")
labels_raw = df['label'].values
images = df.drop('label', axis=1).values

# Normalize and reshape
images = images / 255.0
images = images.reshape(-1, 28, 28, 1)

# Pick a sample
i = 42  # You can change this number to test different ones
sample = images[i]
true_label = labels[labels_raw[i]]

# Predict
prediction = model.predict(sample.reshape(1, 28, 28, 1))
predicted_label = labels[np.argmax(prediction)]

# Show
plt.imshow(sample.reshape(28, 28), cmap='gray', interpolation='bilinear')
plt.title(f"Predicted: {predicted_label}, Actual: {true_label}")
plt.axis('off')
plt.show()
