---
title: 'Understanding Dot Product and Projection in Vectors'
date: '2019-05-20T15:39:50+00:00'
author: gp
layout: post
image: /content/2019/05/projection.jpg
categories:
    - "linear algebra"
math: true
---

In this blog post, we'll dive into two fundamental operations in vector analysis: the
**dot product** and **projection**. We will explain each concept with definitions, 
geometric interpretations, and a practical example.

---

## Dot Product

### What Is It?

Given two vectors  
$$
\mathbf{u} = (u_1, u_2, \dots, u_n) \quad \text{and} \quad \mathbf{v} = (v_1, v_2, \dots, v_n),
$$  
the **dot product** (or **prodotto scalare** in Italian) is defined as:

$$
\mathbf{u} \cdot \mathbf{v} = u_1 v_1 + u_2 v_2 + \dots + u_n v_n.
$$

### Geometric Interpretation

The dot product can also be expressed in terms of the magnitudes of the vectors and the cosine of the angle \(\theta\) between them:

$$
\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \, \|\mathbf{v}\| \cos(\theta).
$$

- When the vectors point in the **same direction**, \(\cos(\theta)\) is positive and the dot product is large.
- When the vectors point in the **same direction**, $(\cos(\theta))$ is positive and the dot product is large.

- When the vectors are **perpendicular**, \(\cos(\theta) = 0\) and the dot product is zero.
- When they point in **opposite directions**, \(\cos(\theta)\) is negative, yielding a negative dot product.

### Sample Calculation

Let’s consider two vectors:

$$
\mathbf{a} = (3, -4) \quad \text{and} \quad \mathbf{b} = (2, 5).
$$

The dot product is calculated as:

$$
\mathbf{a} \cdot \mathbf{b} = (3)(2) + (-4)(5) = 6 - 20 = -14.
$$

This result tells us something about the directional relationship between \(\mathbf{a}\) and \(\mathbf{b}\).

---

## Projection

### What Is It?

The **projection** of one vector onto another helps us determine the component of one vector in the direction of the other. There are two forms of projection:

1. **Scalar Projection (Component):**  
   This tells you the length of the projection (how much one vector extends in the direction of the other). For a vector \(\mathbf{s}\) onto \(\mathbf{r}\), it is given by:
   
  $$
   \text{comp}_{\mathbf{r}}(\mathbf{s}) = \frac{\mathbf{s} \cdot \mathbf{r}}{\|\mathbf{r}\|}.
  $$

2. **Vector Projection:**  
   This gives you the actual vector in the direction of \(\mathbf{r}\) that represents the projection of \(\mathbf{s}\). It is defined as:
   
   $$
   \text{proj}_{\mathbf{r}}(\mathbf{s}) = \left( \frac{\mathbf{s} \cdot \mathbf{r}}{\|\mathbf{r}\|^2} \right) \mathbf{r}.
  $$
   
   Alternatively, it can be written as the scalar projection multiplied by the unit vector in the direction of \(\mathbf{r}\):
   
   $$
   \text{proj}_{\mathbf{r}}(\mathbf{s}) = \text{comp}_{\mathbf{r}}(\mathbf{s}) \cdot \frac{\mathbf{r}}{\|\mathbf{r}\|}.
  $$

![](/content/2019/05/projection.jpg)
_Vector Projection_

### Geometric Insight

- **Scalar Projection:** Provides the magnitude (with sign) of how much of \(\mathbf{s}\) lies along \(\mathbf{r}\).
- **Vector Projection:** Gives the actual vector component along \(\mathbf{r}\).

### Sample Calculation

Consider two vectors in \(\mathbb{R}^2\):

$$
\mathbf{r} = (3, 4) \quad \text{and} \quad \mathbf{s} = (5, 2).
$$

**Step 1: Compute the dot product**

$$
\mathbf{s} \cdot \mathbf{r} = (5)(3) + (2)(4) = 15 + 8 = 23.
$$

**Step 2: Compute the norm of \(\mathbf{r}\)**

$$
\|\mathbf{r}\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5.
$$

**Step 3: Find the scalar projection**

$$
\text{comp}_{\mathbf{r}}(\mathbf{s}) = \frac{23}{5} \approx 4.6.
$$

**Step 4: Find the vector projection**

$$
\text{proj}_{\mathbf{r}}(\mathbf{s}) = \left( \frac{23}{5^2} \right) (3, 4) = \left( \frac{23}{25} \right) (3, 4) \approx (2.76, 3.68).
$$

The vector projection \((2.76, 3.68)\) is the component of \(\mathbf{s}\) that points in the same direction as \(\mathbf{r}\).

---

## Conclusion

In summary:

- **Dot Product:**  
  Combines two vectors to yield a scalar. It reveals information about the magnitude of the vectors and the cosine of the angle between them.

- **Projection:**  
  Decomposes a vector into a component along another vector:
  - The **scalar projection** tells you the magnitude.
  - The **vector projection** provides both the magnitude and direction in the form of a vector.

Understanding these concepts is crucial in many areas, including physics, engineering, and computer graphics, where resolving vectors into components is a common task.

Happy calculating!
