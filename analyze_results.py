import pandas as pd
import ast

def find_and_print_all_measurements(df):
    """
    Finds all rows where energy_error is less than 0.00159 and prints the last
    measurement from the 'total_measurements_at_each_step' column for each.
    """
    threshold = 0.00159
    found_rows = df[df['energy_error'] < threshold]

    if found_rows.empty:
        print(f"No rows found where 'energy_error' is less than {threshold}.")
        return

    print(f"Found {len(found_rows)} rows where 'energy_error' is below {threshold}:")

    # Iterate through all found rows.
    for index, row in found_rows.iterrows():
        try:
            # Parse the string representation of the list and get the last element.
            measurements_list = ast.literal_eval(row['total_measurements_at_each_step'])
            last_measurement = measurements_list[-1]
            print(f"\nRow index: {index}")
            print(f"  - Molecule: {row['molecule']}")
            print(f"  - N_qubits: {row['n_qubits']}")
            print(f"  - Iteration: {row['iteration']}")
            print(f"  - Energy Error: {row['energy_error']:.6f}")
            print(f"  - Last measurement: {last_measurement}")
        except (ValueError, IndexError) as e:
            print(f"\nCould not process the 'total_measurements_at_each_step' value for row index {index}.")
            print(f"  - Error: {e}")

if __name__ == "__main__":
    # Load the CSV file.
    try:
        df = pd.read_csv('beh2/adapt_vqe_bai_qubit_excitation_results_2025-08-1823-12-14_0.005_8.csv')
        find_and_print_all_measurements(df)
    except FileNotFoundError:
        print("The specified CSV file was not found. Please check the file name and path.")
    except Exception as e:
        print(f"An error occurred: {e}")
