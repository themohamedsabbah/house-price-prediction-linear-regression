# House Price Prediction - Linear Regression

This repository implements a multivariate linear regression model from scratch using NumPy to predict house prices based on various features (such as size, number of bedrooms, number of floors, and age).

---

## 📂 Codebase Structure

The project is organized as follows:

```text
house-price-prediction-linear-regression/
├── requirements.txt         # Project dependencies (matplotlib, pandas, scikit-learn, etc.)
├── LICENSE                  # Project license
├── README.md                # Project documentation (this file)
├── src/                     # Source directory
│   ├── main.py              # Main execution script and visualization
│   ├── data_loader.py       # Helper functions to load and validate dataset
│   ├── exceptions.py        # Custom exceptions for robust error handling
│   ├── cost_functions.py    # Compute the model's cost (mean squared error)
│   ├── linear_regression.py # Computes the gradient for parameter optimization
│   └── data/
│       └── houses.txt       # Sample housing dataset (comma-separated value format)
└── tests/                   # Test suite directory
```

---

## 🔍 Function Directory & Locations

Below is a detailed map of all custom classes and functions defined across the codebase, sorted by their host files:

### 📥 `src/data_loader.py`
This module handles reading and validating dataset files.

*   #### `load_dataset`
    *   **Signature**:
        ```python
        def load_dataset(
            path: str | Path,
            *,
            expected_columns: int,
            delimiter: str = ",",
            skiprows: int = 0
        ) -> tuple[np.ndarray, np.ndarray]
        ```
    *   **Description**: Loads a numeric delimited text file (like CSV) into features ($X$) and target ($y$) NumPy arrays.
    *   **Raises**: `DataLoadError` if the file is missing, empty, non-numeric, ragged, or has an unexpected shape.

*   #### `_resolve_path` (Internal)
    *   **Signature**:
        ```python
        def _resolve_path(path: str | Path) -> Path
        ```
    *   **Description**: Resolves path string or object, checking that the target exists, is indeed a file, and is not empty.

---

### ⚠️ `src/exceptions.py`
Houses custom exceptions used throughout the application.

*   #### `DataLoadError`
    *   **Definition**:
        ```python
        class DataLoadError(Exception):
            """Raised when a dataset file cannot be read or fails validation."""
        ```
    *   **Description**: Inherits from standard python `Exception`. Used to encapsulate dataset load failures.

---

### 📐 `src/cost_functions.py`
Computes metrics that quantify model accuracy.

*   #### `compute_cost`
    *   **Signature**:
        ```python
        def compute_cost(X, y, w, b)
        ```
    *   **Description**: Calculates the cost (mean squared error) of the model prediction $f_{w,b}(x) = w \cdot x + b$.
    *   **Parameters**:
        *   `X`: Feature matrix of shape $(m, n)$.
        *   `y`: Target values of shape $(m,)$.
        *   `w`: Model weights of shape $(n,)$.
        *   `b`: Model bias scalar.
    *   **Returns**: Cost float.

---

### 📉 `src/linear_regression.py`
Houses parameter optimization and training math.

*   #### `compute_gradient`
    *   **Signature**:
        ```python
        def compute_gradient(X, y, w, b)
        ```
    *   **Description**: Calculates partial derivatives (gradient) of the cost function with respect to weights $w$ and bias $b$ over all training examples.
    *   **Returns**: A tuple `(dj_db, dj_dw)`:
        *   `dj_db`: Gradient of cost w.r.t. bias (scalar).
        *   `dj_dw`: Gradient of cost w.r.t. weights (NumPy array of shape $(n,)$).

---

### 🚀 `src/main.py`
The project's entry point which coordinates data loading, plotting, and training logic.

*   #### `load_house_data`
    *   **Signature**:
        ```python
        def load_house_data(filename: str = "houses.txt")
        ```
    *   **Description**: Resolves and loads the housing dataset from `src/data/` using `load_dataset`.

---

## 🛠️ Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Execution
Run the main script to load the housing data and visualize the relationships between the features and house prices:
```bash
python src/main.py
```
