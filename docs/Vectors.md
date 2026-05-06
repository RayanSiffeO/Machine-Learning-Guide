# What is a vector?

To know that it is a vector we need to know that it belongs to a vector space, v ∈ ℝⁿ, basically it is an ordered tuple of n real numbers:

    v = (v₁, v₂, ..., vₙ)

In C it is implemented with:

     typedef struct {
        double *data;   // dynamically allocated array of components
        int size;   // n dimension
    } Vector;

![Vector](images/Vectors.svg)

## Vectors in Machine Learning

Understanding vectors in machine learning often seems optional, so people don't give it much importance. However, the truth is quite the opposite. Instead of seeing it as complicated or abstract mathematics, we should see it as a way to represent data and generate algorithms that work.
Therefore, I will give you an overview of what vectors are used for and how they influence the work environment.

## Why vectors really matter

In ML we have to keep in mind that everything becomes vectors, for example when we have the features of a user, it is a vector of characteristics (age, login days, etc...):

**User = (age, login days)** its a feature vector representing the user

For other types of data such as text, word vectors are used with methods such as embeddings, which transform a word into a numeric vector.
(For example: **apple -> (0.2, -0.5, 1.1, ...)**)

From this perspective, we can more easily see how ML models work. Many algorithms check how similar two data points are, using techniques such as Euclidean distances or dot products (which will be explained later). Many models are also built directly with vector calculations such as the famous linear regressor or neural networks, although they are used for model optimization (for example, the use of gradients).

Okay, now that we have a simple explanation of why we use vectors, let's get started with the mathematical and computational explanations. If you already know some of the topics, I'll provide an index of everything I'll explain so you can organize your thoughts and find what interests you.

## Index

### 1. Vector Creation & Memory Management
- `vector_create()` — Allocate a new vector
- `vector_free()` — Free memory
- `vector_copy()` — Deep copy of a vector
- `vector_from_array()` — Create from existing array
- `vector_fill()` — Initialize all elements to a value
- `vector_zero()` — Create a zero vector

### 2. Element Access & Modification
- `vector_get()` — Read element at index
- `vector_set()` — Write element at index
- `vector_slice()` — Extract a portion of the vector
- `vector_concat()` — Combine two vectors

### 3. Basic Vector Operations
- `vector_add()` — Element-wise addition
- `vector_sub()` — Element-wise subtraction
- `vector_scale()` — Multiply by scalar
- `vector_shift()` — Add scalar to all elements
- `vector_mul_elem()` — Element-wise multiplication (Hadamard product)
- `vector_div_elem()` — Element-wise division
- `vector_axpy()` — Optimized: v = v + α·u (used in numerical methods)

### 4. Dot Product & Similarity (Core to ML)
- `vector_dot()` — Dot product (inner product)
- **`vector_cosine_similarity()`** — Cosine distance (how aligned two vectors are)
- **`vector_euclidean_distance()`** — Euclidean distance (straight-line distance)
- **`vector_manhattan_distance()`** — Manhattan distance (L1 norm distance)

### 5. Norms & Magnitudes
- `vector_norm()` — L2 norm (Euclidean length)
- `vector_norm1()` — L1 norm (Manhattan length)
- `vector_norm_inf()` — Infinity norm (max absolute value)
- `vector_normalize()` — Scale to unit length (norm = 1)

### 6. Statistical Operations
- `vector_sum()` — Sum of all elements
- `vector_mean()` — Average value
- `vector_variance()` — Variance (spread of data)
- `vector_std()` — Standard deviation
- `vector_min()` — Minimum value
- `vector_max()` — Maximum value
- `vector_argmin()` — Index of minimum
- `vector_argmax()` — Index of maximum

### 7. Element-wise Mathematical Operations
- `vector_abs()` — Absolute value
- `vector_sqrt()` — Square root
- `vector_pow()` — Power (v^p)
- `vector_log()` — Natural logarithm
- `vector_exp()` — Exponential (e^v)
- `vector_clip()` — Clamp values between bounds

### 8. Activation Functions (Neural Networks)
- `vector_sigmoid()` — Sigmoid activation (1/(1+e^-v))
- `vector_tanh()` — Hyperbolic tangent
- `vector_relu()` — ReLU (max(0, v))
- `vector_leaky_relu()` — Leaky ReLU with α
- `vector_softmax()` — Softmax (normalized exponentials, for classification)

### 9. Categorical Operations
- `vector_one_hot()` — One-hot encoding (class_idx → binary vector)

### 10. Utilities & Debugging
- `vector_check_dims()` — Verify dimension compatibility
- `vector_print()` — Display vector contents
- `vector_print_named()` — Display with label


## 1. Vector Creation & Memory Management

### Create a vector (vector_create())

To learn how to create vectors, we first need to know what we want to represent. This could be usernames or user bank debts, but the important thing is to have a mental map of the features we want. After that, we need to convert them into numbers and organize them into a single vector. So you can see it better:

- username = "Julio" ->  (9, -32, 0.1) debt = 2450$  ----> Vector x = (9, -32, 0.1, 0.2450) 


