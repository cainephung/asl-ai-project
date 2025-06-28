import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV with correct relative path
df = pd.read_csv("data/sign_mnist_train/sign_mnist_train.csv")

# Pick one sample
sample = df.iloc[0]

# Get label and image
label = sample.iloc[0]
image = sample.iloc[1:].to_numpy().reshape(28, 28)

# Display
plt.title(f"Label: {chr(label + 65)}")  # Convert 0-25 to A-Z
plt.imshow(image, cmap='gray', interpolation='bilinear')
plt.axis("off")
plt.show()
