---
id: 227
title: 'Training a Keras Model on MNIST: A Simple, Modular Approach'
date: '2017-07-25T09:43:57+00:00'
author: 'Gian Paolo'
layout: post
guid: 'https://genmind.ch/?p=227'
permalink: /training-a-keras-model-on-mnist-a-simple-modular-approach/
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
image: /wp-content/uploads/2025/03/mnist_sample_digits.png
categories:
    - keras
    - 'Machine Learning'
---

In today’s post, I’m excited to share a beginner-friendly guide on how to train a neural network using Keras on the famous MNIST dataset. Whether you’re new to deep learning or just looking for a clear example, this guide breaks everything down into manageable pieces with methods and a main function to keep things organized.

## What’s the MNIST Dataset?

The MNIST dataset is a collection of 70,000 handwritten digits (split into training and testing sets) that’s been a go-to benchmark for machine learning experiments. In our example, we’ll use **TensorFlow Datasets** (tfds) to load the dataset and build an efficient data pipeline.

## Breaking Down the Code

### 1. Loading and Preprocessing Data

We start by loading the MNIST dataset with `tfds.load()`, which not only retrieves the data but also gives us useful metadata. We define a helper function called `normalize_img` to convert the image pixel values from integers (0-255) to floating-point numbers between 0 and 1. This simple normalization is key for faster model convergence.

Then, we build our training and testing pipelines:

- **Training Pipeline:** We apply normalization, cache the data for speed, shuffle the training examples to ensure a good mix, and then batch and prefetch the data.
- **Testing Pipeline:** Similar to training but without shuffling, so we can reliably measure our model’s performance.

### 2. Building the Model

Next, we define a `create_model` function that constructs a basic feed-forward neural network:

- **Flatten Layer:** Converts the 28×28 image into a flat vector.
- **Dense Layer:** Adds a fully-connected layer with 128 neurons and a ReLU activation function.
- **Output Layer:** Produces 10 outputs corresponding to the digits 0 through 9.

We then compile the model using the Adam optimizer and the sparse categorical crossentropy loss function. This setup is standard for classification problems like MNIST.

### 3. Training the Model

The `train_model` function is responsible for running the training process. It calls `model.fit()`, feeding in our training dataset and using the testing dataset for validation over a set number of epochs (in our example, 6 epochs).

### 4. Orchestrating with the Main Function

Finally, our `main` function ties everything together. It loads the data, creates the model, and starts the training process. This modular structure not only keeps our code clean but also makes it easier to modify or extend later.

## Full Code Example

```

import tensorflow as tf
import tensorflow_datasets as tfds

def load_and_preprocess_data(batch_size=128):
    """
    Loads the MNIST dataset and creates training and testing pipelines.
    
    Returns:
        ds_train: Training dataset with normalization, shuffling, and batching.
        ds_test: Testing dataset with normalization and batching.
        ds_info: Dataset metadata (contains info like number of examples).
    """
    # Load the MNIST dataset as supervised (image, label) tuples along with dataset info.
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,            # Shuffle file order (useful for large datasets)
        as_supervised=True,            # Returns (image, label) tuples
        with_info=True,                # Includes metadata about the dataset
    )
    
    def normalize_img(image, label):
        """
        Normalizes images: converts from uint8 to float32 and scales to [0, 1].
        """
        return tf.cast(image, tf.float32) / 255., label

    # Prepare the training dataset pipeline:
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()  # Cache the dataset in memory for faster access
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)  # Prefetch to improve latency

    # Prepare the testing dataset pipeline:
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()  # Cache for performance
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    
    return ds_train, ds_test, ds_info

def create_model():
    """
    Creates a basic feed-forward neural network model for MNIST.
    
    Returns:
        A compiled tf.keras.Sequential model.
    """
    model = tf.keras.models.Sequential([
        # Flatten the 28x28 image into a 784-dimensional vector.
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        # Dense layer with 128 neurons and ReLU activation.
        tf.keras.layers.Dense(128, activation='relu'),
        # Output layer with 10 neurons (one per MNIST digit).
        tf.keras.layers.Dense(10)
    ])
    
    # Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return model

def train_model(model, ds_train, ds_test, epochs=6):
    """
    Trains the model on the training dataset and validates it on the test dataset.
    
    Args:
        model: The compiled Keras model.
        ds_train: Preprocessed training dataset.
        ds_test: Preprocessed testing dataset.
        epochs: Number of training epochs.
    """
    model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test,
    )

def main():
    """
    Main function that orchestrates the data loading, model creation, and training process.
    """
    # Load and preprocess the MNIST dataset.
    ds_train, ds_test, ds_info = load_and_preprocess_data()
    
    # Create the model.
    model = create_model()
    
    # Train the model with the prepared datasets.
    train_model(model, ds_train, ds_test)

# Entry point of the script
if __name__ == '__main__':
    main()

```

## Wrapping Up

This example demonstrates a clean and modular approach to building a neural network with Keras. By breaking down the process into distinct functions—data preparation, model creation, and training—we keep the code organized and easier to maintain. I hope you enjoyed this simple walkthrough and feel more confident in setting up your own projects!

Feel free to ask any questions or share your experiences in the comments below. Happy coding!