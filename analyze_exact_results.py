import pandas as pd
import ast

def sum_measurements_until_threshold(df):
    """
    Sums the first estimated measurement until the energy_error falls below a threshold.
    """
    threshold = 0.00159
    total_sum = 0

    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        # Check if the energy_error has fallen below the threshold
        if row['energy_error'] < threshold:
            print(f"The energy error falls below {threshold} at row index {index}, iteration {row['iteration']}.")
            print(f"The cumulative sum of the first estimated measurements is: {total_sum}")
            return

        # If not, add the first estimated measurement to the total sum
        try:
            # Handle the specific string format by removing brackets and splitting by space.
            measurements_str = row['Estiamted measurements [0.001, 0.01, 0.1]'].strip('[]')
            measurements_list = measurements_str.split()
            first_measurement = measurements_list[0]
            total_sum += float(first_measurement)
        except (ValueError, IndexError, TypeError) as e:
            print(f"Warning: Could not process 'Estiamted measurements [0.001, 0.01, 0.1]' for row index {index}. Skipping row. Error: {e}")

    # If the loop completes without meeting the condition
    print(f"The energy error did not fall below {threshold} within the provided data.")
    print(f"The total sum of the first estimated measurements is: {total_sum}")

if __name__ == "__main__":
    # Load the CSV file.
    try:
        df = pd.read_csv('beh2/adapt_vqe_qubit_excitation_results_2025-08-1814-53-54_exact_estimates.csv')
        sum_measurements_until_threshold(df)
    except FileNotFoundError:
        print("The specified CSV file was not found. Please check the file name and path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
