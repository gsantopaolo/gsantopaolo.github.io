---
id: 238
title: 'Training a scikit‐learn Model on MNIST: A Simple, Modular Approach'
date: '2017-07-27T12:14:42+00:00'
author: gp
layout: post
guid: 'https://genmind.ch/?p=238'
permalink: /training-a-scikit%e2%80%90learn-model-on-mnist-a-simple-modular-approach/
site-sidebar-layout:
    - default
ast-site-content-layout:
    - default
site-content-style:
    - default
site-sidebar-style:
    - default
theme-transparent-header-meta:
    - default
astra-migrate-meta-layouts:
    - set
image: /content/2025/03/minist_scikit.png
categories:
    - 'Machine Learning'
tags:
    - MNIST
---

In this post, I’ll walk you through building a simple machine learning pipeline to classify handwritten digits from the MNIST dataset using scikit‐learn. If you’re looking for a beginner-friendly and modular approach, this guide will break everything down into manageable pieces—from data loading and preprocessing to model training and evaluation.

---

## What’s the MNIST Dataset?

The MNIST dataset is a benchmark collection of 70,000 handwritten digits. It’s split into a training set and a test set, and each image is 28×28 pixels in size. This dataset has become a standard for testing and comparing machine learning algorithms.

---

## Data Loading and Preprocessing

We start by fetching the MNIST data from OpenML using scikit‐learn’s built-in function. Then, we split the dataset into training and testing sets and apply standard scaling. Normalizing the data (zero mean and unit variance) is important for faster convergence and improved performance.

```

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"].astype(np.int8)

# Split into training and test sets (e.g., 60k for training, 10k for testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42
)

# Normalize the data: scale features to zero mean and unit variance.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

```

## Building the Model

For our classifier, we’ll use Logistic Regression from scikit‐learn. This model is easy to implement and interpret while providing competitive accuracy on MNIST. (Feel free to experiment with other models like Support Vector Machines or K-Nearest Neighbors!)

```

from sklearn.linear_model import LogisticRegression

# Create a logistic regression classifier
clf = LogisticRegression(
    solver='saga',    # efficient for large datasets
    penalty='l2',     # L2 regularization (ridge)
    max_iter=100,     # increase iterations if necessary for convergence
    C=1.0,            # Inverse regularization strength
    random_state=42
)

```

---

## Training the Model

Next, we train the model using our preprocessed training data. The `fit()` method does all the heavy lifting behind the scenes.

```

# Train the classifier on the training data
clf.fit(X_train, y_train)

```

## Evaluation and Visualization

Once the model is trained, we evaluate its performance on the test set by checking the accuracy and visualizing a confusion matrix. You can also inspect individual predictions and even display some of the misclassified images.

```

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Evaluate the model on the test data
test_accuracy = clf.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate a classification report
print(classification_report(y_test, clf.predict(X_test)))

# Compute and display the confusion matrix
cm = confusion_matrix(y_test, clf.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

```

You might also want to visualize a few test samples alongside their predicted labels:

```

# Display a few test images with their predictions
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
preds = clf.predict(X_test[:5])
for ax, image, prediction in zip(axes, X_test[:5], preds):
    ax.imshow(image.reshape(28, 28), cmap='gray', interpolation='nearest')
    ax.set_title(f"Predicted: {prediction}")
    ax.axis('off')
plt.suptitle("Sample Predictions")
plt.show()

```

## Wrapping Up

In this post, we demonstrated a modular approach to training a scikit‐learn model on the MNIST dataset. We:

- **Loaded and preprocessed the data:** using `fetch_openml`, splitting into training and testing sets, and applying standard scaling.
- **Built a simple logistic regression model:** to classify the handwritten digits.
- **Trained and evaluated the model:** achieving competitive accuracy and visualizing the results with a confusion matrix and sample predictions.

This approach makes it easy to swap out components (e.g., trying different classifiers or preprocessing steps) and is a great starting point for more advanced experimentation.

Feel free to leave comments or questions below. Happy coding!
