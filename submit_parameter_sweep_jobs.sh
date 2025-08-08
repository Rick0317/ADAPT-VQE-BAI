#!/bin/bash

# Script to submit multiple ADAPT-VQE jobs with parameter sweeping
# Each job will test different combinations of x (0.001 to 0.01) and y (1 to 10) parameters

echo "Submitting ADAPT-VQE parameter sweep jobs..."

# Define the base parameters (same as in submit_individual_jobs.sh)
declare -a BASE_PARAMS=(
    "lih_fer.bin lih 12 4 uccsd 1024"
    "beh2_fer.bin beh2 14 4 uccsd 8192"
)

# Define the x parameter range (0.001 to 0.01 with 0.001 steps)
declare -a X_VALUES=(0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01)

# Define the y parameter range (1 to 10)
declare -a Y_VALUES=(1 2 3 4 5 6 7 8 9 10)

# Counter for job numbering
job_counter=0

# Submit jobs for each base parameter combination
for base_param in "${BASE_PARAMS[@]}"; do
    # Parse base parameters
    read -r mol_file mol n_qubits n_electrons pool_type shots <<< "$base_param"
    
    echo "Processing base parameters: $mol with $pool_type pool"
    
    # Submit jobs for each x, y combination
    for x_val in "${X_VALUES[@]}"; do
        for y_val in "${Y_VALUES[@]}"; do
            job_counter=$((job_counter + 1))
            
            echo "Submitting job $job_counter: $mol with x=$x_val, y=$y_val"

            # Create temporary individual job script
            cat > temp_job_${job_counter}.sh << EOF
#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=23:30:00
#SBATCH --job-name=adapt_${mol}_x${x_val}_y${y_val}
#SBATCH --output=adapt_${mol}_x${x_val}_y${y_val}_%j.out
#SBATCH --error=adapt_${mol}_x${x_val}_y${y_val}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ricky.huang@mail.utoronto.ca

# Load required modules
module load python
module load conda3
source ~/adapt-vqe-env/bin/activate

# Set up matplotlib config directory
export MPLCONFIGDIR=\$SLURM_TMPDIR/matplotlib
mkdir -p \$MPLCONFIGDIR

# Set memory and CPU optimizations
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "SLURM job started at \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Running $mol with parameters: $mol_file $n_qubits $n_electrons $pool_type $shots x=$x_val y=$y_val"

echo "Testing environment..."
which python
python --version
python -c "import numpy as np; import qiskit; print('NumPy version:', np.__version__); print('Qiskit version:', qiskit.__version__)"

# Print available memory and CPU info
echo "Available memory:"
free -h
echo "CPU info:"
lscpu | grep "CPU(s):"

cd \$SLURM_SUBMIT_DIR

# Create unique log filename
LOG_FILE="adapt_vqe_${mol}_x${x_val}_y${y_val}_\${SLURM_JOB_ID}.log"

echo "Starting ADAPT-VQE parameter sweep for $mol with x=$x_val, y=$y_val..."
echo "Working directory: \$(pwd)"

# Run the ADAPT-VQE script with parameter sweep
python -u adapt_vqe_exact_bai_scipy_minimization_multi_params.py "$mol_file" "$mol" "$n_qubits" "$n_electrons" "$pool_type" "$shots" $x_val $y_val 2>&1 | tee -a "\$LOG_FILE"

# Check exit status
if [ \$? -eq 0 ]; then
    echo "ADAPT-VQE for $mol with x=$x_val, y=$y_val completed successfully at \$(date)"
else
    echo "ADAPT-VQE for $mol with x=$x_val, y=$y_val failed with exit code \$? at \$(date)"
fi

# Print final memory usage
echo "Final memory usage:"
free -h

# Copy results to a timestamped directory
TIMESTAMP=\$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results_${mol}_x${x_val}_y${y_val}_\${TIMESTAMP}_\${SLURM_JOB_ID}"
mkdir -p \$RESULTS_DIR

# Copy output files
echo "Copying results for $mol with x=$x_val, y=$y_val..."
cp *${mol}*x${x_val}*y${y_val}*.csv \$RESULTS_DIR/ 2>/dev/null || echo "No CSV files found for $mol-x${x_val}-y${y_val}"
cp *${mol}*.json \$RESULTS_DIR/ 2>/dev/null || echo "No JSON cache files found for $mol"
cp "\$LOG_FILE" \$RESULTS_DIR/ 2>/dev/null || echo "No log file found"

# Also copy any generic output files
cp *.csv \$RESULTS_DIR/ 2>/dev/null || echo "No additional CSV files in root directory"
cp *.json \$RESULTS_DIR/ 2>/dev/null || echo "No additional JSON cache files in root directory"

echo "Results copied to: \$RESULTS_DIR"
echo "Job finished at \$(date)"
EOF

            # Submit this individual job
            JOB_ID=$(sbatch temp_job_${job_counter}.sh | awk '{print $4}')
            echo "  → Submitted as job ID: $JOB_ID"

            # Clean up temporary script
            rm temp_job_${job_counter}.sh

            # Small delay to avoid overwhelming the scheduler
            sleep 1
        done
    done
done

# Calculate total number of jobs
total_jobs=$(( ${#BASE_PARAMS[@]} * ${#X_VALUES[@]} * ${#Y_VALUES[@]} ))

echo ""
echo "All $total_jobs jobs submitted successfully!"
echo "Parameter combinations:"
echo "  - Base parameters: ${#BASE_PARAMS[@]} combinations"
echo "  - X values: ${#X_VALUES[@]} values (${X_VALUES[0]} to ${X_VALUES[-1]})"
echo "  - Y values: ${#Y_VALUES[@]} values (${Y_VALUES[0]} to ${Y_VALUES[-1]})"
echo "  - Total: $total_jobs jobs"
echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo ""
echo "To cancel all your jobs:"
echo "  scancel -u \$USER"
echo ""
echo "To monitor specific parameter combinations:"
echo "  squeue -u \$USER | grep 'adapt_lih_x0.005_y5'"
echo "  squeue -u \$USER | grep 'adapt_beh2_x0.003_y3'" 