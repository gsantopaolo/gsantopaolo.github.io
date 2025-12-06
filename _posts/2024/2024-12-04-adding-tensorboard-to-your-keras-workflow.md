---
title: 'Adding TensorBoard to Your Keras Workflow'
date: '2024-12-04T22:08:55+00:00'
author: gp
layout: post
permalink: /adding-tensorboard-to-your-keras-workflow/
image: /content/2025/03/tensorboard.jpg
categories:
    - 'Machine Learning'
tags:
    - 'Data Visualization'
---

Adding TensorBoard as a data visualization tool to your Keras model is easier than you might think. With just a few extra lines of code, you can gain valuable insights into your model’s training process.

For this example I’m going to use an old post [Training a Keras Model on MNIST: A Simple, Modular Approach](https://genmind.ch/training-a-keras-model-on-mnist-a-simple-modular-approach/) , the code is still valid, by adding a few line of code we will have TensorBoard seamlessly integrated.

#### What You Need to Do

1. **Set Up a Log Directory:**  
    Create a directory (e.g., `logs/mnist`) where TensorBoard will store its logs.
    
    ```python
    LOG_DIR = os.path.join(os.getcwd(), 'logs', 'mnist')
    ```
2. **Add the TensorBoard Callback:**  
    Import TensorBoard from Keras and instantiate it with your log directory. Then, pass this callback  
    xxxxx  
    to the `model.fit()` function. This small change allows you to log training metrics, weight histograms, and even your model graph.
    
    ```python
        # Create a TensorBoard callback to monitor training progress
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR,  # Directory where TensorBoard logs (scalars, histograms, graphs) will be stored.
            histogram_freq=1  # Logs weight and activation histograms every epoch.
        )
    
        model.fit(
            ds_train,
            epochs=epochs,
            validation_data=ds_test,
            callbacks=[tensorboard_callback]  # Add TensorBoard callback
        )
    ```
3. **Visualize Your Metrics:**  
    Once your training begins, run the command
    
    ```bash
    tensorboard --logdir=logs/mnist
    ```

Et voilà, in your terminal to launch the TensorBoard web interface. Here, you can monitor the progress of your training.

<figure aria-describedby="caption-attachment-268" class="wp-caption aligncenter" id="attachment_268" style="width: 691px">![](content/2025/03/tensorboard-300x123.jpg)<figcaption class="wp-caption-text" id="caption-attachment-268">TensorBoard</figcaption></figure>#### How Does TensorBoard Compare to Weights &amp; Biases?

While TensorBoard is built into the TensorFlow ecosystem and provides a fast, no-cost solution for basic visualization needs, it can be limited when it comes to advanced experiment tracking. On the other hand, [Weights &amp; Biases](https://wandb.ai/?utm_source=genmind.ch) (W&amp;B) integrates seamlessly with TensorBoard logging but goes further by offering:

- **Enhanced Experiment Tracking:** Automatic logging of hyperparameters, code versions, and artifacts.
- **Robust Comparison Tools:** Easy filtering and grouping of experiments for quick comparisons.
- **Collaboration Features:** Streamlined sharing and team collaboration through its cloud-based dashboards.

In short, if you’re looking for a straightforward visualization tool for individual projects, TensorBoard is a great fit. However, for larger experiments and team-based workflows, W&amp;B might be worth the extra setup.

---

This post quickly shows how a small code change can significantly boost your model tracking capabilities.  
I hope you find this simple introduction helpful. As always, <del>you can download the full source code here</del> \[edit\] I moved all the code on my [GitHub repo](https://github.com/gsantopaolo/dataviz?utm_source=genmind.ch)  
Happy coding!
