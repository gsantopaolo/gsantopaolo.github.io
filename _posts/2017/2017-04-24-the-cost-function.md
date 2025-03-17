---
title: 'The Cost Function'
date: '2017-04-24T15:46:34+00:00'
author: gp
layout: post
categories:
    - 'Machine Learning'
tags:
    - 'cost function'
math: true
---

$$  
J(w,b) = \\frac{1}{2m} \\sum\_{i=1}^m \\left(f\_{w,b}\\bigl(x^{(i)}\\bigr) – y^{(i)}\\right)^2  
$$

The cost function in linear regression is a way to measure how well a model’s predictions match the actual target values. In this context, we are trying to find parameters \\(w\\) and \\(b\\) that make the predicted output \\(\\hat{y}^{(i)}\\) as close as possible to the true target \\(y^{(i)}\\) for every training example. Here’s a detailed breakdown of how it works:

**1. Error Calculation:**  
For each training example $i$, the error is computed as the difference between the predicted value and the actual value:  
$$error^{(i)} = \\hat{y}^{(i)} – y^{(i)}$$  
This error represents how far off the prediction is from the target.

**2. Squaring the Error:**  
The error for each training example is squared to ensure that both positive and negative differences contribute positively to the total error. Squaring also penalizes larger errors more heavily. For each example, this is:  
$$(\\hat{y}^{(i)} – y^{(i)})^2 $$

**3. Summing Over All Examples:**  
To get a complete picture of the model’s performance, the squared errors for all training examples are summed up:  
$$  
\\sum\_{i=1}^m \\left( \\hat{y}^{(i)} – y^{(i)} \\right)^2  
$$  
where \\( m \\) is the total number of training examples.

**4. Averaging the Error:**  
If we simply summed the squared errors, a larger training set would naturally yield a larger number. To standardize this measure, the sum is divided by \\( m \\) to compute the mean squared error (MSE):  
$$  
\\frac{1}{m} \\sum\_{i=1}^m \\left( \\hat{y}^{(i)} – y^{(i)} \\right)^2  
$$

**5. Division by 2 (Convention):**  
By convention, the cost function is often defined with an extra division by 2:  
$$  
J(w, b) = \\frac{1}{2m} \\sum\_{i=1}^m ( \\left(\\hat{y}^{(i)} – y^{(i)} \\right)^2  
$$  
This extra factor of 1/2 is introduced so that when you take the derivative of the cost function with respect to the model parameters (w and b), the constant 2 from the exponent cancels out, simplifying the gradient calculations.

**6. Purpose of the Cost Function:**  
The goal of the Cost Function in linear regression is to minimize the cost function J over parameters w and b.  
In training, the goal is to adjust the parameters ($w$ and $b$) to minimize $J(w, b)$. This minimization is usually performed using optimization techniques like gradient descent.
