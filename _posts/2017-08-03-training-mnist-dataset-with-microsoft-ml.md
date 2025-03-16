---
id: 218
title: 'Training MNIST dataset with Microsoft.ML'
date: '2017-08-03T10:59:20+00:00'
author: 'Gian Paolo'
layout: post
guid: 'https://genmind.ch/?p=218'
permalink: /training-mnist-dataset-with-microsoft-ml/
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
image: /wp-content/uploads/2017/08/mlnet.jpg
categories:
    - 'Machine Learning'
tags:
    - MNIST
---

This project was is about training a model on the classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/) — a collection of handwritten digits that’s been a staple in the ML community. Let’s take a friendly tour through the code and ideas behind this experiment.

## Setting Up the Experiment

The goal was simple: train a model that can recognize handwritten digits. To achieve this, I used Microsoft.ML, a powerful library that brings machine learning into the .NET ecosystem. The project starts by loading the MNIST dataset, which was stored as CSV files for both training and testing. If you’re new to the MNIST dataset, it’s essentially a grid of pixel values that represent numbers from 0 to 9.

## Breaking Down the Code

### Data Handling

I created a simple data structure (`MNIST_Data`) to map the CSV data. Each record contains a label (the digit) and an array of pixel values. To make data reading easier, I employed <a data-end="1139" data-start="1088" rel="noopener" target="_new">CsvHelper</a> to load the CSV file into a list of objects. This helper method abstracts away a lot of the manual parsing, letting me focus on the machine learning side of things.

```
 
public class MNIST_Data
{
    public MNIST_Data()
    {
        Pixels = new byte[784]; // Each image is 28x28 pixels.
    }

    [Column("0", name: "Number")]
    public byte Number;

    [Column("1", name: "Pixels")]
    public byte[] Pixels;

    [Column(ordinal: "2", name: "Label")]
    public float Label;
}

```

### Building the Pipeline

Next comes the fun part—building the machine learning pipeline. Microsoft.ML’s pipeline concept allows you to chain together data processing and training steps. In this example, I set up a pipeline that reads the data and then applies the **AveragedPerceptronBinaryClassifier**. Even though this is a binary classifier, it was a great starting point to understand how to configure and train a model.

```
 
var pipeline = new LearningPipeline();
var dataSource = CollectionDataSource.Create(helper.ReadMNIST_Data(trainingDataLocation));
pipeline.Add(dataSource);
pipeline.Add(new AveragedPerceptronBinaryClassifier());
```

### Training and Evaluating

Once the pipeline was built, training the model was just a matter of invoking the `Train` method. After training, I used another helper class—`ModelEvaluator`—to test the model’s accuracy against the test data. The evaluation provided insight into how well the model was able to recognize handwritten digits.

```
 
var perceptronBinaryModel = new ModelBuilder(trainingDataLocation, new AveragedPerceptronBinaryClassifier()).BuildAndTrain();
var perceptronBinaryMetrics = modelEvaluator.Evaluate(perceptronBinaryModel, testDataLocation);
```

This process, while straightforward, encapsulates the core idea of machine learning: preparing data, training a model, and then evaluating its performance.

## Reflections and Modern Thoughts

Looking back, this project was a fantastic learning experience. It demonstrated how even a few lines of code can bring a machine learning model to life using the Microsoft.ML framework. Although the library has evolved considerably since then, the foundational concepts remain the same. Today, you might explore more advanced models and tools, but the joy of seeing your first ML model in action is something I’ll always cherish.

Whether you’re just starting out or are already deep into machine learning, I hope this little trip down memory lane inspires you to experiment and build. The world of ML is vast, and with tools like Microsoft.ML, integrating these powerful techniques into your applications has never been more accessible.

Source code available on my [github repo](https://github.com/gsantopaolo/MNIST_Sample)

Happy coding and keep exploring!