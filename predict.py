import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def ft_validate_data(data: pd.DataFrame) -> None:
    """
    Validate the input data for required columns and data types.
    """
    # Check required columns
    required_columns = ['km', 'price']
    missing_columns = [col for col in required_columns if
                       col not in data.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}")

    # Check for null values
    if data.isnull().any().any():
        raise ValueError("Dataset contains null values")

    # Check for negative values
    if (data['km'] < 0).any():
        raise ValueError("Dataset contains negative mileage values")

    if (data['price'] < 0).any():
        raise ValueError("Dataset contains negative price values")

    # Check if dataset is empty
    if len(data) == 0:
        raise ValueError("Dataset is empty")


def ft_read_data(file_path: str) -> pd.DataFrame:
    """
    Read and validate data from CSV file.
    """
    try:
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        data = pd.read_csv(file_path)
        ft_validate_data(data)
        return data

    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty")
    except pd.errors.ParserError:
        raise ValueError("Invalid CSV format")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")


def ft_load_params(filename: str) -> tuple[float, float]:
    """
    Load trained model parameters from a text file.
    """
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Parameters file not found: {filename}")
        with open(filename, 'r') as f:
            lines = f.readlines()

            if len(lines) != 2:
                raise ValueError("Invalid parameter file format")

            try:
                theta0 = float(lines[0].strip())
                theta1 = float(lines[1].strip())
            except ValueError:
                raise ValueError("Invalid parameter values in file")

            return theta0, theta1
    except Exception as e:
        raise ValueError(f"Error loading parameters: {str(e)}")


def plot_data_and_model(X, y, theta):
    """Plot the training data and the linear regression model"""
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, label="Training data", color="blue")
    plt.plot(X, theta[0] + theta[1] * X, label="Model", color="red")
    plt.xlabel("Kilometers")
    plt.ylabel("Price")
    plt.title('Car Prices vs km')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def plot_user_prediction(km: float, price: float) -> None:
    """
    Plot the user's prediction point along with training data and model line.

    Args:
        km (float): Kilometer value entered by user
        price (float): Predicted price for given kilometers
    """
    try:
        data = ft_read_data('data.csv')
        theta0, theta1 = ft_load_params('model_params.txt')
        plt.figure(figsize=(10, 6))
        plt.scatter(data['km'], data['price'],
                    color='blue', alpha=0.5,
                    label='Training data')
        km_range = np.array([0, data['km'].max()])
        prices = theta0 + theta1 * km_range
        plt.plot(km_range, prices, 'r-', label='Model')

        # Tracer le point de prédiction
        plt.scatter([km], [price],
                    color='green', s=100, marker='*',
                    label='Your prediction')

        # Personnaliser le graphique
        plt.title('Car Price Prediction')
        plt.xlabel('Kilometers')
        plt.ylabel('Price (€)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Ajouter une annotation pour le point de prédiction
        plt.annotate(f'Prediction:\n{price:.2f}€',
                     xy=(km, price),
                     xytext=(10, 10),
                     textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow',
                               alpha=0.5),
                     arrowprops=dict(arrowstyle='->'))

        plt.show()

    except Exception as e:
        print(f"Error displaying visualization: {str(e)}")


def ft_predict() -> None:
    """
    Interactive function to predict car prices based on user input km.
    Uses trained model parameters to make predictions.
    """
    try:
        theta0, theta1 = ft_load_params('model_params.txt')

        while True:
            try:
                km_input = input("\nEnter km (or 'q' to quit): ")
                if km_input.lower() == 'q':
                    break

                if not km_input.replace('.', '').isdigit():
                    raise ValueError("Please enter a valid number")

                km = float(km_input)
                if km < 0:
                    raise ValueError("Mileage cannot be negative")

                if km > 400000:  # Reasonable upper limit
                    raise ValueError("Mileage value out of range")
                price = theta1 * km + theta0
                price = max(0, int(price))
                if np.isnan(price) or np.isinf(price):
                    raise ValueError("Invalid prediction value")
                print(f"\nFor {km}km, estimated price is: {price:.2f}€")
                plot_user_prediction(km, price)
            except Exception as e:
                print(f"Error: {e}")

    except Exception as e:
        print(f"Error: {e}")


def main():
    """
    Main function to run the price prediction program.
    Loads data, displays visualization, and starts prediction interface.
    """
    ft_predict()


if __name__ == "__main__":
    main()
