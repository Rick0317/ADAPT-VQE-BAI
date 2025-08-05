import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

if __name__ == "__main__":
    # Read both CSV files
    df1 = pd.read_csv('adapt_vqe_intermediate_h4_uccsd_results_2025-08-0400-58.csv')
    df2 = pd.read_csv('adapt_vqe_intermediate_h4_uccsd_results_2025-08-0501-21.csv')

    # Function to extract data from a dataframe
    def extract_data(df):
        energy_errors = df['energy_error'].values

        # Extract the last element from total_measurements_at_each_step column
        total_measurements = []
        for measurements_str in df['total_measurements_at_each_step']:
            # Parse the string representation of the list
            measurements_list = ast.literal_eval(measurements_str)
            # Get the last element
            total_measurements.append(measurements_list[-1])

        total_measurements = np.array(total_measurements)

        # Extract estimated measurements for different accuracies
        estimated_measurements = []
        for est_str in df['Estiamted measurements [0.001, 0.01, 0.1]']:
            # Parse the string representation of the numpy array
            # Remove brackets and split by spaces, then convert to float
            est_str_clean = est_str.strip('[]')
            est_values = [float(x) for x in est_str_clean.split()]
            estimated_measurements.append(est_values)

        estimated_measurements = np.array(estimated_measurements)

        # Calculate cumulative measurements for each accuracy level
        cumulative_measurements_001 = np.cumsum(estimated_measurements[:, 0])  # 0.001 accuracy

        return energy_errors, total_measurements, cumulative_measurements_001

    # Extract data from both files
    energy_errors1, total_measurements1, cumulative_measurements_001_1 = extract_data(df1)
    energy_errors2, total_measurements2, cumulative_measurements_001_2 = extract_data(df2)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Define chemical accuracy threshold (1.6 mHa = 0.0016 Ha)
    chemical_accuracy = 0.0016

    # Add light blue shaded area below chemical accuracy threshold
    plt.axhspan(0, chemical_accuracy, alpha=0.2, color='lightblue', label='Chemical Accuracy Region')

    # Plot BAI data from both files
    plt.loglog(total_measurements1, energy_errors1, 'bo-', linewidth=2, markersize=8, label='BAI results')
    plt.loglog(cumulative_measurements_001_2, energy_errors2, 'co-', linewidth=2, markersize=8, label='Estimate (0.001 accuracy)')

    # Add chemical accuracy horizontal line
    plt.axhline(y=chemical_accuracy, color='red', linestyle='--', linewidth=2, label=f'Chemical Accuracy ({chemical_accuracy:.4f} Ha)')

    # Add labels and title
    plt.xlabel('Total Measurements', fontsize=12)
    plt.ylabel('Energy Error (Ha)', fontsize=12)
    plt.title('Energy Error vs Total Measurements in Gradient Selection for H4 UCCSD', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    # Add iteration numbers as annotations for the main line
    for i, (x, y) in enumerate(zip(total_measurements1, energy_errors1)):
        plt.annotate(f'{i}', (x, y), textcoords="offset points", xytext=(0,10),
                    ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('energy_error_vs_measurements.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print the data points for verification
    print("Run 1 - Data points (iteration, actual_measurements, energy_error):")
    for i, (meas, err) in enumerate(zip(total_measurements1, energy_errors1)):
        print(f"Iteration {i}: {meas:,} measurements, {err:.6f} error")

    print("\nRun 2 - Data points (iteration, cumulative_estimated_measurements, energy_error):")
    for i, (meas, err) in enumerate(zip(cumulative_measurements_001_2, energy_errors2)):
        print(f"Iteration {i}: {meas:,.0f} measurements, {err:.6f} error")
