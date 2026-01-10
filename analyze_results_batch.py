import pandas as pd
import ast
import os
import glob

def find_and_extract_first_measurement(df, filename):
    """
    Finds the first row where energy_error is less than 0.00159 and extracts the
    'total_measurements' value from that row.
    Returns the total_measurements value from the first qualifying row.
    """
    threshold = 0.00159
    found_rows = df[df['energy_error'] < threshold]

    if found_rows.empty:
        print(f"No rows found in {filename} where 'energy_error' is less than {threshold}.")
        return None

    print(f"Found {len(found_rows)} rows in {filename} where 'energy_error' is below {threshold}:")

    # Get the total_measurements value from the first qualifying row
    first_row = found_rows.iloc[0]
    total_measurements = first_row['total_measurements']
    print(f"  Using total_measurements from row {first_row.name}: {total_measurements}")
    return total_measurements

def categorize_files(filenames):
    """
    Categorize files based on their names
    """
    categories = {
        'qubit_excitation': [],
        'qubit_pool': [],
        'uccsd': [],
        'other': []
    }

    for filename in filenames:
        if 'qubit_excitation' in filename:
            categories['qubit_excitation'].append(filename)
        elif 'qubit_pool' in filename:
            categories['qubit_pool'].append(filename)
        elif 'uccsd' in filename:
            categories['uccsd'].append(filename)
        else:
            categories['other'].append(filename)

    return categories

def process_csv_files(file_pattern, output_suffix):
    """
    Process CSV files based on pattern and save results with given suffix
    """
    beh2_dir = "h2o/"
    all_measurements = []
    file_results = {}
    categorized_results = {
        'qubit_excitation': {},
        'qubit_pool': {},
        'uccsd': {},
        'other': {}
    }

    # Get all CSV files in h4/ directory
    csv_files = glob.glob(os.path.join(beh2_dir, "*.csv"))

    # Filter files based on pattern
    if file_pattern == "exclude_exact_estimates":
        filtered_files = [f for f in csv_files if not f.endswith("_exact_estimates.csv")]
        print(f"Found {len(filtered_files)} CSV files to process (excluding _exact_estimates.csv files)")
    elif file_pattern == "only_exact_estimates":
        filtered_files = [f for f in csv_files if f.endswith("_exact_estimates.csv")]
        print(f"Found {len(filtered_files)} CSV files to process (only _exact_estimates.csv files)")
    else:
        filtered_files = csv_files
        print(f"Found {len(filtered_files)} CSV files to process (all files)")

    # Categorize files
    file_categories = categorize_files([os.path.basename(f) for f in filtered_files])

    print("=" * 80)
    print("File categorization:")
    for category, files in file_categories.items():
        print(f"  {category}: {len(files)} files")
    print("=" * 80)

    for csv_file in filtered_files:
        filename = os.path.basename(csv_file)
        print(f"\nProcessing: {filename}")
        print("-" * 50)

        try:
            df = pd.read_csv(csv_file)
            first_measurement = find_and_extract_first_measurement(df, filename)

            if first_measurement is not None:
                file_results[filename] = first_measurement
                all_measurements.append(first_measurement)

                # Categorize the results
                if 'qubit_excitation' in filename:
                    categorized_results['qubit_excitation'][filename] = first_measurement
                elif 'qubit_pool' in filename:
                    categorized_results['qubit_pool'][filename] = first_measurement
                elif 'uccsd' in filename:
                    categorized_results['uccsd'][filename] = first_measurement
                else:
                    categorized_results['other'][filename] = first_measurement

                print(f"  Extracted first measurement {first_measurement} from {filename}")
            else:
                print(f"  No measurement extracted from {filename}")

        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    # Save results to CSV file
    output_file = f"h2o_first_measurements{output_suffix}.csv"

    # Prepare data for CSV
    csv_data = {}
    max_length = 0

    # Add categorized results
    for category, files in categorized_results.items():
        if files:  # Only include categories that have files
            category_measurements = []
            for filename, measurement in files.items():
                category_measurements.append(int(measurement))  # Convert numpy int64 to regular int
            csv_data[category] = category_measurements
            max_length = max(max_length, len(category_measurements))

    # Add all measurements
    csv_data['all'] = [int(x) for x in all_measurements]  # Convert numpy int64 to regular int
    max_length = max(max_length, len(all_measurements))

    # Write to CSV
    with open(output_file, 'w') as f:
        # Write header
        f.write(','.join(csv_data.keys()) + '\n')

        # Write data rows
        for i in range(max_length):
            row = []
            for category in csv_data.keys():
                if i < len(csv_data[category]):
                    row.append(str(csv_data[category][i]))
                else:
                    row.append('')  # Empty cell for shorter lists
            f.write(','.join(row) + '\n')

    print(f"\n" + "=" * 80)
    print(f"Results saved to: {output_file}")
    print(f"Total first measurements extracted: {len(all_measurements)}")
    print(f"Files processed: {len(file_results)}")

    # Print summary by category
    print(f"\nSUMMARY BY CATEGORY:")
    print("-" * 30)
    for category, files in categorized_results.items():
        if files:
            category_total = len(files)
            print(f"{category}: {len(files)} files, {category_total} first measurements")

    return all_measurements, file_results, categorized_results

if __name__ == "__main__":
    print("Processing files excluding _exact_estimates.csv...")
    print("=" * 80)
    all_measurements_regular, file_results_regular, categorized_results_regular = process_csv_files("exclude_exact_estimates", "")

    print("\n\nProcessing files with _exact_estimates.csv...")
    print("=" * 80)
    all_measurements_exact, file_results_exact, categorized_results_exact = process_csv_files("only_exact_estimates", "_exact_estimates")

    print("\n\nOVERALL SUMMARY:")
    print("=" * 80)
    print(f"Regular files processed: {len(file_results_regular)}")
    print(f"Exact estimates files processed: {len(file_results_exact)}")
    print(f"Total files processed: {len(file_results_regular) + len(file_results_exact)}")
