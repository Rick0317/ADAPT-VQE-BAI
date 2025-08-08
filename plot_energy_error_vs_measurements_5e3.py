import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast


def parse_total_measurements(measurements_str):
    """Parse the total_measurements_at_each_step string and return the last element"""
    try:
        # Remove brackets and split by spaces, then convert to float
        measurements_str = measurements_str.strip('[]')
        measurements = [float(x) for x in measurements_str.split(", ")]
        return measurements[-1]  # Return the last element
    except Exception as e:
        print(f"Error parsing measurements string: {measurements_str}")
        print(f"Error: {e}")
        return 0


def plot_energy_error_vs_measurements(csv_filename, csv_filename_exact):
    """Plot energy_error vs total_measurements using last element from total_measurements_at_each_step"""

    # Read the CSV file
    df = pd.read_csv(csv_filename)

    # Extract energy errors
    energy_errors = df['energy_error'].values

    # Parse total measurements (last element from each row)
    total_measurements = []
    for measurements_str in df['total_measurements_at_each_step']:
        last_measurement = parse_total_measurements(measurements_str)
        print(last_measurement)
        total_measurements.append(last_measurement)

    # Convert to numpy array
    total_measurements = np.array(total_measurements)

    """Plot energy_error vs total_measurements using estimated measurements data"""

    # Read the CSV file
    df_exact = pd.read_csv(csv_filename_exact)

    # Extract energy errors
    energy_errors_exact = df_exact['energy_error'].values

    # Parse estimated measurements for each row
    estimated_measurements_list = []
    for measurements_str in df_exact['Estiamted measurements [0.001, 0.01, 0.1]']:
        measurements = parse_estimated_measurements(measurements_str)
        estimated_measurements_list.append(measurements)

    # Convert to numpy array for easier manipulation
    estimated_measurements_array = np.array(estimated_measurements_list)

    # Calculate cumulative measurements for each accuracy level
    cumulative_measurements_001 = np.cumsum(
        estimated_measurements_array[:, 0])  # 0.001 accuracy

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot for each accuracy level
    plt.loglog(cumulative_measurements_001, energy_errors_exact, 'ro-', linewidth=2,
               markersize=8, label='0.001 accuracy estimates')


    # Add some data points annotations
    for i in range(0, len(energy_errors_exact), 2):  # Annotate every other point
        plt.annotate(f'{i}',
                     (cumulative_measurements_001[i], energy_errors_exact[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Plot energy error vs total measurements
    plt.loglog(total_measurements, energy_errors, 'bo-', linewidth=2,
               markersize=8, label='BAI method')

    # Add chemical accuracy line and shaded region
    chemical_accuracy = 0.0016  # Ha
    plt.axhline(y=chemical_accuracy, color='red', linestyle='--', linewidth=2,
                label='Chemical Accuracy')
    plt.axhspan(0, chemical_accuracy, alpha=0.3, color='lightblue',
                label='Chemical Accuracy Region')

    # Customize the plot
    plt.xlabel('Total Measurements', fontsize=14)
    plt.ylabel('Energy Error (Ha)', fontsize=14)
    plt.title(
        'Energy Error vs Total Measurements',
        fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add iteration number annotations
    for i in range(0, len(energy_errors), 2):  # Annotate every other point
        plt.annotate(f'{i}',
                     (total_measurements[i], energy_errors[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.tight_layout()
    plt.show()

    # Print some statistics
    print(f"Dataset: {csv_filename}")
    print(f"Number of iterations: {len(energy_errors)}")
    print(f"Final energy error: {energy_errors[-1]:.6f} Ha")
    print(f"Final total measurements: {total_measurements[-1]:.2e}")

    # Check if chemical accuracy is achieved
    if energy_errors[-1] <= chemical_accuracy:
        print(f"✓ Chemical accuracy achieved!")
    else:
        print(
            f"✗ Chemical accuracy not yet achieved. Need {chemical_accuracy - energy_errors[-1]:.6f} Ha improvement.")

    # Print all data points for verification
    print("\nData points:")
    print("Iteration | Energy Error | Total Measurements")
    print("-" * 45)
    for i in range(len(energy_errors)):
        print(
            f"{i:9d} | {energy_errors[i]:11.6f} | {total_measurements[i]:15.2e}")


def parse_estimated_measurements(measurements_str):
    """Parse the estimated measurements string and return as numpy array"""
    try:
        # Remove brackets and split by spaces, then convert to float
        measurements_str = measurements_str.strip('[]')
        measurements = [float(x) for x in measurements_str.split()]
        return np.array(measurements)
    except Exception as e:
        print(f"Error parsing measurements string: {measurements_str}")
        print(f"Error: {e}")
        return np.array([0, 0, 0])

if __name__ == "__main__":
    # Plot the data from your CSV file
    csv_filename = "adapt_vqe_intermediate_lih_uccsd_results_2025-08-0602-23.csv"
    csv_filename_exact = "adapt_vqe_intermediate_lih_uccsd_results_2025-08-0514-14_exact.csv"

    plot_energy_error_vs_measurements(csv_filename, csv_filename_exact)
