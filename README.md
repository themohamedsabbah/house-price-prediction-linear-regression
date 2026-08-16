# Multi-Variable Linear Regression for House Price Prediction

A clean, robust, and highly educational implementation of **Multi-Variable Linear Regression** built from scratch in Python using **NumPy** and **Matplotlib**. This project demonstrates the fundamental concepts of machine learning—specifically, custom **gradient descent**, **cost function computation**, and the critical importance of **z-score feature normalization** when dealing with multi-variable datasets of varying scales.

Inspired by the classical housing dataset from Andrew Ng's Machine Learning specialization, this repository is fully documented, type-safe, and includes modular code for data loading, training, error handling, and rich visualizations.

---

## 📖 Table of Contents
1. [Project Structure](#-project-structure)
2. [Dataset Overview](#-dataset-overview)
3. [Mathematical Foundation](#-mathematical-foundation)
    - [Linear Regression Model](#1-linear-regression-model)
    - [Mean Squared Error Cost Function](#2-mean-squared-error-cost-function)
    - [Gradient Descent Updates](#3-gradient-descent-updates)
    - [Z-Score Feature Normalization](#4-z-score-feature-normalization)
4. [Module Descriptions](#-module-descriptions)
5. [Installation & Setup](#-installation--setup)
6. [Interactive Exploration](#-interactive-exploration)
7. [Training & Convergence Analysis](#-training--convergence-analysis)
    - [Without Feature Normalization](#1-without-feature-normalization)
    - [With Feature Normalization](#2-with-feature-normalization)
8. [License](#-license)

---

## 📁 Project Structure

The project has been architected with strong software engineering practices, separating data parsing, exception handling, algorithmic implementation, and visualization:

```text
house-price-prediction-linear-regression/
├── .gitignore
├── .python-version
├── LICENSE
├── pyproject.toml              # Project dependencies & metadata
├── uv.lock                    # Lockfile for reproducible builds
├── README.md                  # This documentation
├── src/
│   ├── data_loader.py         # Type-safe utility to load & validate housing data
│   ├── exceptions.py          # Custom domain exceptions (DataLoadError)
│   ├── cost_functions.py      # Vectorized and loop-based cost computation
│   ├── linear_regression.py   # Core gradient descent, gradient computation, and normalization
│   ├── plot.py                # Specialized plotting routines (distributions, cost histories)
│   ├── visual_steps.ipynb     # Interactive Jupyter Notebook stepping through the process
│   └── data/
│       └── houses.txt         # Raw housing dataset (comma-delimited)
└── tests/                     # Reserved directory for unit tests
```

---

## 📊 Dataset Overview

The housing dataset (`src/data/houses.txt`) consists of **100 tabular samples**. Each row contains **4 features** describing a house and a **target variable** representing the house's sale price:

| Feature Name | Description | Approximate Scale / Range |
| :--- | :--- | :--- |
| **Size** | Total indoor living space in square feet | $500\text{ sqft} - 3,000\text{ sqft}$ |
| **Bedrooms** | Number of bedrooms | $1 - 5$ bedrooms |
| **Floors** | Number of floors | $1 - 3$ floors |
| **Age** | Age of the home in years | $0 - 100\text{ years}$ |
| **Price (Target)**| Sale price of the home in **$1,000s** (e.g., 271.5 = $271,500) | $\$150k - \$800k$ |

### Feature Scale Mismatch
Because the features represent vastly different physical quantities, their numerical ranges differ by orders of magnitude (e.g., size ranges in the thousands, while floor count is typically 1 or 2). This massive scale mismatch causes standard gradient descent to struggle or diverge unless the features are normalized.

---

## 🧮 Mathematical Foundation

### 1. Linear Regression Model
A multi-variable linear model predicts the target value $f_{\mathbf{w}, b}(\mathbf{x})$ for a given input vector $\mathbf{x} = [x_0, x_1, \dots, x_{n-1}]^T$ as:

$$f_{\mathbf{w}, b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b = w_0 x_0 + w_1 x_1 + \dots + w_{n-1} x_{n-1} + b$$

where:
*   $\mathbf{w}$ is the weight vector of size $n$ (one weight per feature).
*   $b$ is the scalar bias parameter.

### 2. Mean Squared Error Cost Function
To measure how well our model fits the training data, we compute the Mean Squared Error (MSE) cost function $J(\mathbf{w}, b)$ over $m$ training examples:

$$J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=0}^{m-1} \left( f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)} \right)^2$$

where $\mathbf{x}^{(i)}$ and $y^{(i)}$ represent the $i$-th training feature vector and target value respectively.

### 3. Gradient Descent Updates
To minimize the cost $J(\mathbf{w}, b)$, the model parameters are updated iteratively in the opposite direction of the gradient:

$$w_j \leftarrow w_j - \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \qquad \text{for } j = 0, \dots, n-1$$

$$b \leftarrow b - \alpha \frac{\partial J(\mathbf{w},b)}{\partial b}$$

where $\alpha$ is the learning rate, and the gradients are calculated as:

$$\frac{\partial J(\mathbf{w},b)}{\partial w_j} = \frac{1}{m} \sum_{i=0}^{m-1} \left( f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)} \right) x^{(i)}_j$$

$$\frac{\partial J(\mathbf{w},b)}{\partial b} = \frac{1}{m} \sum_{i=0}^{m-1} \left( f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)} \right)$$

### 4. Z-Score Feature Normalization
Because features have vastly different ranges, the contours of the cost function $J(\mathbf{w},b)$ form an extremely elongated, narrow valley. Gradient descent bounces back and forth and can only converge with a microscopic learning rate $\alpha$.

We solve this using **z-score feature normalization**, which transforms each feature so that it has a mean ($\mu$) of $0$ and a standard deviation ($\sigma$) of $1$:

$$x'_j = \frac{x_j - \mu_j}{\sigma_j}$$

where:
*   $\mu_j = \frac{1}{m} \sum_{i=0}^{m-1} x^{(i)}_j$ is the mean of feature $j$.
*   $\sigma_j = \sqrt{\frac{1}{m} \sum_{i=0}^{m-1} (x^{(i)}_j - \mu_j)^2}$ is the standard deviation of feature $j$.

With normalized features, the cost function contours are nearly spherical, allowing gradient descent to take large, direct steps towards the global minimum using a much higher learning rate.

---

## 🧩 Module Descriptions

### 1. Data Loader & Exceptions (`src/data_loader.py` & `src/exceptions.py`)
Provides a safe, type-hinted wrapper around `numpy.loadtxt` to read housing files. It enforces defensive constraints, raising a custom `DataLoadError` if:
*   The data file is missing, empty, or non-numeric.
*   The columns in the dataset do not match the expected width (features + target = 5).

### 2. Cost Computation (`src/cost_functions.py`)
Implements the core Mean Squared Error formula.
```python
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2 * m)
    return cost
```

### 3. Training & Normalization (`src/linear_regression.py`)
Contains the optimization logic:
*   `compute_gradient`: Computes $\frac{\partial J}{\partial \mathbf{w}}$ and $\frac{\partial J}{\partial b}$ in a single pass.
*   `gradient_descent_houses`: Iterates gradient updates, dynamically logging progress and saving parameter histories at specified safe intervals to avoid memory bloat.
*   `zscore_normalize_features`: Normalizes columns to zero-mean and unit-variance, returning the normalized matrix along with original means ($\mu$) and standard deviations ($\sigma$) required for inference on new raw inputs.

### 4. Visualization Utilities (`src/plot.py`)
*   `norm_plot`: Plots distributions of raw features and normalized features, overlaying a red bell-curve representing fitted Gaussian probability density functions (PDFs) to visualize normalizations.
*   `plot_cost_i_w`: Plots Cost vs Iteration to verify convergence, alongside Cost vs Weight parameter slices.

---

## ⚙️ Installation & Setup

Make sure you have Python (>= 3.10) installed. This project is configured to use [uv](https://github.com/astral-sh/uv)—a fast Python package installer and resolver—but standard `pip` can also be used.

### Option A: Using `uv` (Recommended)
1. Clone this repository to your machine.
2. Initialize and install dependencies:
   ```bash
   uv sync
   ```
3. Run the interactive notebook using the synced environment:
   ```bash
   uv run jupyter notebook src/visual_steps.ipynb
   ```

### Option B: Using standard `pip`
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install matplotlib pandas notebook numpy scipy
   ```
3. Start the Jupyter notebook server:
   ```bash
   jupyter notebook src/visual_steps.ipynb
   ```

---

## 📓 Interactive Exploration

The `src/visual_steps.ipynb` notebook is an interactive guide demonstrating the entire training pipeline:
1. **Loading Data**: Loads and validates the dataset.
2. **Raw Visualizations**: Plots the correlation between raw house characteristics and prices.
3. **Training on Raw Data**: Shows how gradient descent struggles to converge when learning rates are tiny (e.g., $10^{-7}$).
4. **Applying Normalization**: Calculates means and standard deviations, normalizes raw variables, and compares feature distributions before and after normalization.
5. **Fast Convergence**: Runs gradient descent with $\alpha = 0.1$ and observes instantaneous cost minimization.
6. **Inference / Prediction**: Uses learned weights and bias parameters to predict house values and overlays predicted trends against actual labels.

---

## 📈 Training & Convergence Analysis

### 1. Without Feature Normalization
When attempting to run gradient descent on raw data, the algorithm is extremely sensitive. 
*   If $\alpha = 9.9 \times 10^{-7}$ (a tiny step size), the cost drops slowly but requires millions of iterations to converge.
*   If $\alpha = 1 \times 10^{-6}$ (only slightly larger), the steps along the "Size" gradient are too large, and the cost immediately **explodes (diverges)** to infinity.

```text
Iter: 0    Cost: 9.31e+04    w_0: 0.54    w_1: 0.00    w_2: 0.00    w_3: 0.01    b: 0.00
Iter: 1    Cost: 1.22e+05    w_0:-0.08    w_1:-0.00    w_2:-0.00    w_3:-0.00    b: 0.00
...
w, b found by gradient descent: w: [-0.75, -0.001, -0.0008, -0.024], b: -0.00
```
*Result:* Failed convergence due to scale disparity.

### 2. With Feature Normalization
After applying Z-Score normalization, all inputs have a peak-to-peak range of similar magnitude. This lets us use a massive learning rate of **$\alpha = 0.1$**. 

```text
Iter: 0      Cost: 5.73e+04    w_0:  8.9e+00    w_1:  3.0e+00    w_2:  3.3e+00    w_3: -6.0e+00    b:  3.6e+01
Iter: 100    Cost: 2.21e+02    w_0:  1.1e+02    w_1: -2.0e+01    w_2: -3.1e+01    w_3: -3.8e+01    b:  3.6e+02
Iter: 200    Cost: 2.19e+02    w_0:  1.1e+02    w_1: -2.1e+01    w_2: -3.3e+01    w_3: -3.8e+01    b:  3.6e+02
...
w, b found by gradient descent:
w: [110.613, -21.473, -32.660, -37.779], b: 362.24
```
*Result:* Fast, stable convergence in fewer than 200 iterations with a final cost of **~219.71**.

### Model Interpretation (Normalized Space)
With a learned bias of $b = 362.24$, a house with average parameters ($\mathbf{x}' = [0, 0, 0, 0]$) has a baseline price of **$362.24k ($362,240)**. 
*   **Size weight ($w_0 = 110.61$)**: Increasing house size by 1 standard deviation increases price by **$110.61k**.
*   **Bedrooms weight ($w_1 = -21.47$)**: Additional bedrooms (holding size constant) slightly reduces price (reflecting smaller room sizes or specific market factors in the training subset).
*   **Age weight ($w_3 = -37.78$)**: Increasing the home age by 1 standard deviation reduces its value by **$37.78k**.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute this code for learning and development.
