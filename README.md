# Machine-Learning Guide
A **high-performance, numerically stable Machine Learning library written in pure C11**. From SVD decomposition to Random Forests, all with transparent mathematical documentation.

## Project Objective

- **The goal of this library is to provide a transparent, high-performance, and educational implementation of Machine Learning algorithms from scratch.**
- **While modern libraries often act as "black boxes," this project focuses on:**
- **Mathematical Clarity: Code that maps directly to linear algebra and optimization formulas.**
- **Numerical Stability: Utilizing advanced techniques like SVD to handle ill-conditioned data where standard methods fail.**
- **Hardware Intimacy: Demonstrating how manual memory management and pointer arithmetic can be leveraged for efficient numerical computing.**
- **Knowing which ML model to use in different situations, thus gaining the ability to make informed decisions when working with data**

## Features

## Implemented ML models

- **Linear Regression** (SVD-based, numerically stable)
- **Logistic Regression** (with Adam/SGD/RMSprop optimizers)
- **Random Forest** (classification & regression, OOB scoring)

 ## Core knowledge for learning from the repository

 - **Matrix & Vector** (Not only the calculations, but also the geometric implications, thus providing a clear spatial understanding of vectors and matrices, which are the heart of machine learning)
- **Memory Management** (Although it's not mandatory and you can skip it, it's always useful for any engineer to understand memory management, even if they don't use it in their work.)
- **Numerical algorithms** (Methods such as SVD, gradient descent, or decision trees are essential for understanding how ML models work, so it's always good to include them.)
- **Machine Learning Models** (Knowing how they work internally is core knowledge for anyone in data science; it's not just about applying a .fit, but understanding the math behind it to create coherent solutions.)

## Statistical validations

- **Cross-Validation** (Methods such as K-Fold, Stratified K-Fold, and LOO will be explained. We will explain how our model validations work. Although I consider them important, for the time being they will not have their own document but will be included within the ML documentation until I deem it necessary.)
- **Residual Diagnostics** (Even if a model performs well (high R², good accuracy, etc.), it may be poorly constructed. Residual Diagnostics help detect this. That's why I think it's a good idea to include a document that explains methods like Shapiro-Wilk, Durbin-Watson, Breusch-Pagan.)
- **Metrics** (I think everyone is familiar with machine learning model metrics, but to provide an overview for those who know nothing about machine learning, I will also explain basic metrics like R², RMSE, MAE, AUC, Precision, Recall, and F1 in a separate document.)
- **Numerical Stability Analysis** (We will also use a range of techniques to check whether your model's calculations are mathematically reliable or whether small changes in the data could lead to large errors in the results. (condition number, rank detection))

## Acknowledgments & References

- Numerical Recipes 3rd Edition (The Art of Scientific Computing): Although many people say it's an outdated book and that its methods are sometimes ineffective, as theoretical documentation it's unrivaled. The explanations are relentless, and I recommend it to anyone interested in the low level computation.

- 3Blue1Brown (Grant Sanderson): (His videos on linear algebra, vectors, and methods like SVD not only provide a theoretical explanation but also show you how everything works dimensionally—things that university only teaches you to calculate and then stop. He offers a great introduction to linear algebra for anyone interested.)

- Visual Kernel: (A YouTube channel with four videos on linear algebra that helped me understand methods like SVD. Perfect for visualizations and spatial reasoning, besides providing a pretty decent understanding of how it works.)

- Claude and Notebook LM: (Honestly, Haiku 4.5 is the best tool you can use to get links to forums and documents. Then you can add them to Notebook LM and you have quality information that has helped me a lot. Claude also helped me solve several code bugs with Sonnet. It's truly a great tool for programming and studying without having to struggle with libraries that don't have books or navigate through countless forums without any answers.)

-  Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (Aurélien Géron): (This book is the best for real-world production; it teaches you tools you'll use every day in the workplace. But most importantly, it explains all the key concepts and theory about machine learning that most people ignore. It's a powerful resource for understanding the world of data science.)

## Project Structure

``` 

Numerical-ML-in-C/
│
├── 📄 CMakeLists.txt              # CMake build configuration
├── 📄 LICENSE                     # Project license
├── 📄 README.md                   # Project documentation
│
├── src/                        # Source code (canonical)
│   ├── core/                   # Core data structures & utilities
│   │   ├── Matrix.c / Matrix.h    # Matrix operations
│   │   ├── memory.c / memory.h    # Memory management utilities
│   │   └── vector.c / vector.h    # Vector operations
│   │
│   ├── models/                 # ML model implementations
│   │   ├── LinearRegression.c / .h    # Linear Regression model
│   │   ├── LogisticRegression.c / .h  # Logistic Regression model
│   │   ├── RandomForest.c / .h        # Random Forest classifier
│   │   └── argsort.c / argsort.h      # Sorting utility (used by RF)
│   │
│   └── numerical/              # Numerical methods & algorithms
│       ├── gradient.c / gradient.h    # Gradient descent / optimization
│       ├── random.c / random.h        # Random number generation
│       ├── SVD.c / SVD.h              # Singular Value Decomposition
│       └── tree.c / tree.h            # Decision tree structure
│
├── examples/                   # Usage examples
│   ├── LinearTest.c               # Linear Regression example
│   ├── LogisticTest.c             # Logistic Regression example
│   └── RandomForestTest.c         # Random Forest example
│
├── datasets/                   # Sample CSV datasets
│   ├── dataset_binario.csv        # Binary classification dataset
│   ├── dataset_clasificacion.csv  # Multi-class classification dataset
│   └── dataset_regresion.csv      # Regression dataset
│
├── build/                      # Precompiled artifacts & build copies
│   ├── linux/                  # Linux precompiled libraries
│   │   ├── libregresion.lib
│   │   └── logistic.lib
│   │
│   ├── windows/                # Windows precompiled libraries
│   │   ├── linear.dll
│   │   └── logistic.dll
│   │
│   ├── Models/                 # Build copies of model sources
│   │   ├── LinearRegression.c / .h
│   │   ├── LogisticRegression.c / .h / .obj
│   │   ├── RandomForest.c / .h
│   │   └── argsort.c / argsort.h
│   │
│   └── numerical/              # Build copies of numerical sources
│       ├── gradient.c / gradient.h
│       ├── random.c / random.h
│       ├── SVD.c / SVD.h
│       └── tree.c / tree.h
│
├── docs/                       # Documentation (directory)
└── tests/                      # Tests (directory)
```
 
---

## Contributions

If you find any errors in the documentation or bugs in the code, I would greatly appreciate it if you submitted a pull request so I can fix them. I'll take the time to check them.

## Requirements
- C11 compiler (GCC/Clang)
- CMake 3.10+
- No external dependencies (pure C)

## (WARNING!!)Assumptions & Limitations
 
### Mathematical Assumptions
- **Linear Regression**: Errors are i.i.d. N(0, σ²) *(for CI only, not point estimates)*
- **Logistic Regression**: Binary classification (multiclass via one-vs-rest if needed)
- **Random Forest**: Classification or regression, but not both in one model

### Numerical Assumptions
- **Input data**: Finite double-precision floats (no NaN/Inf handling)
- **Dimensions**: m ≥ n for regression (more samples than features)
- **Matrix rank**: Full rank assumed for inversion (SVD regularizes)

### Limitations
- Single-threaded (no OpenMP/CUDA)
- No sparse matrix support
- CSV I/O not included (use Python/R for preprocessing)
- Classification limited to discrete integer labels