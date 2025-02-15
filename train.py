import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    """Load training data from CSV file"""
    try:
        data = pd.read_csv('data.csv')
        # need to reshape for matrix operations
        X = data['km'].to_numpy().reshape(-1, 1)
        y = data['price'].to_numpy().reshape(-1, 1)
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def validate_data(X, y):
    """Validate input data"""
    if X is None or y is None:
        return False
    if len(X) == 0 or len(y) == 0:
        print("Error: Empty dataset")
        return False
    if np.isnan(X).any() or np.isnan(y).any():
        print("Error: Dataset contains NaN values")
        return False
    return True


def normalize_data(X):
    """Normalize features using z-score normalization"""
    try:
        mean = np.mean(X)
        std = np.std(X)
        if std == 0:
            raise ValueError("Standard deviation is zero - cannot normalize")
        X_norm = (X - mean) / std
        return X_norm, mean, std
    except Exception as e:
        print(f"Error in normalization: {e}")
        return None, None, None


def add_ones_column(X):
    """Add a column of ones to the feature matrix"""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def gradient_descent(X, y, n_iteration, learning_rate=0.01):
    """
    Perform gradient descent with matrix operations
    X: feature matrix with bias column added (mx2)
    y: target vector (mx1)
    Returns: Final theta values and loss history
    """
    m = X.shape[0]
    theta = np.zeros((2, 1))
    prev_cost = float('inf')
    cost_history = np.zeros(n_iteration)

    for i in range(0, n_iteration):
        # model : X * theta (avec dimensions: mx2 * 2x1 = mx1)
        predictions = X.dot(theta)

        # algo gradient descent: (1/m) * X^T * (predictions - y)
        gradients = (1 / m) * X.T.dot(predictions - y)

        # Update parameters
        theta = theta - learning_rate * gradients

        # fonction cout
        current_cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)

        cost_history[i] = current_cost

        # Check convergence
        if abs(prev_cost - current_cost) < 1e-6:
            print(f"Converged")
            break

        prev_cost = current_cost

    return theta, cost_history


def plot_data_and_model(X, y, theta, cost_history):
    """Plot the training data and the linear regression model and cost history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.scatter(X, y, label="Training data", color="blue")
    ax1.plot(X, theta[0] + theta[1] * X, label="Model", color="red")
    ax1.set_xlabel("Kilometers")
    ax1.set_ylabel("Price")
    ax1.set_title('Car Prices vs km')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(cost_history, color='green')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Cost")
    ax2.set_title('Cost History')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def calculate_accuracy(X, y, theta):
    """Calculate model accuracy using RMSE, R² score, and prediction accuracy"""
    predictions = theta[0] + theta[1] * X
    mse = np.mean((predictions - y) ** 2)
    rmse = np.sqrt(mse)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - predictions) ** 2)
    r2_score = 1 - (ss_residual / ss_total)

    mean_y = np.mean(y)
    accuracy = (1 - (rmse / mean_y)) * 100

    print("Model Performance Metrics:")
    print(
        f"  - Mean Squared Error (MSE): {mse:.2f} "
        f"(Écart moyen au carré entre prédictions et valeurs réelles)")
    print(
        f"  - Root Mean Squared Error (RMSE): {rmse:.2f} "
        f"(Erreur moyenne en unités de prix)")
    print(
        f"  - R² Score: {r2_score:.4f}"
        f" (Qualité d'ajustement, proche de 1 signifie un bon modèle)")
    print(
        f"  - Model Accuracy: {accuracy:.2f}% "
        f"(Précision du modèle basée sur l'écart type des prix)")


def save_parameters(theta0, theta1):
    """save parameters in model_params.txt"""
    try:
        with open('model_params.txt', 'w') as f:
            f.write(f"{theta0}\n{theta1}")
        print("Training completed. Parameters saved to model_params.txt")
    except IOError as e:
        print(f"Error saving to file: {e}")
        raise


def train_model():
    """Main training function"""

    X, y = load_data()
    if not validate_data(X, y):
        return

    X_norm, X_mean, X_std = normalize_data(X)
    if X_norm is None:
        return
    y_norm, y_mean, y_std = normalize_data(y)
    if X_norm is None:
        return

    X_norm = add_ones_column(X_norm)

    theta_norm, cost_history = gradient_descent(X_norm, y_norm, 500)

    # Denormalize
    theta1 = theta_norm[1][0] * (y_std / X_std)
    theta0 = theta_norm[0][0] * y_std + y_mean - theta1 * X_mean
    theta = np.array([theta0, theta1])

    save_parameters(theta0, theta1)

    plot_data_and_model(X, y, theta, cost_history)
    calculate_accuracy(X, y, theta)


if __name__ == "__main__":
    train_model()
