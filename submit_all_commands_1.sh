#!/bin/bash

# Script to submit multiple individual ADAPT-VQE jobs for each command in notes.txt
# Each command will be run 10 times with separate job IDs

echo "Submitting individual ADAPT-VQE jobs for each command in notes.txt..."
echo "Each command will be run 10 times..."

# Read commands from notes.txt
if [ ! -f "notes.txt" ]; then
    echo "Error: notes.txt not found!"
    exit 1
fi

# Counter for total jobs
total_jobs=0

# Process each line in notes.txt
while IFS= read -r line; do
    # Skip empty lines
    if [[ -z "$line" ]]; then
        continue
    fi

    # Parse the command line
    # Format: python script.py mol_file mol n_qubits n_electrons pool_type shots [other_params...]
    read -r script_name mol_file mol n_qubits n_electrons pool_type shots accuracy radius target_accuracy <<< "$line"

    echo "Processing command: $script_name with $mol ($pool_type pool)"

    # Submit 5 jobs for this command (testing)
    for run in {1..5}; do
        total_jobs=$((total_jobs + 1))

        echo "  Submitting run $run/5 for $mol with $pool_type pool (Job $total_jobs total)"

        # Set the virtual environment path
        VENV_PATH="/scratch/rick0317/adapt_vqe_env"

        # Create temporary individual job script
        cat > temp_job_${mol}_${pool_type}_run${run}.sh << EOF
#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=23:30:00
#SBATCH --job-name=${script_name}_${mol}_${pool_type}_run${run}
#SBATCH --output=${script_name}_${mol}_${pool_type}_run${run}_%j.out
#SBATCH --error=${script_name}_${mol}_${pool_type}_run${run}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ricky.huang@mail.utoronto.ca

# Load required modules for Trillium
module load StdEnv/2023 gcc python/3.11 symengine/0.11.2

# Activate existing virtual environment
echo "Environment variables debug:"
echo "SCRATCH: \$SCRATCH"
echo "SLURM_TMPDIR: \$SLURM_TMPDIR"
echo "USER: \$USER"
echo ""

SCRATCH_ENV="$VENV_PATH"
echo "Activating existing virtual environment: \$SCRATCH_ENV"

if [ ! -d "\$SCRATCH_ENV" ]; then
    echo "Error: Virtual environment not found at \$SCRATCH_ENV"
    echo "Please create the environment first with:"
    echo "  python -m venv \$SCRATCH_ENV"
    echo "  source \$SCRATCH_ENV/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

source "\$SCRATCH_ENV/bin/activate"
if [ \$? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

echo "✓ Virtual environment activated successfully"

# Set up matplotlib config directory
export MPLCONFIGDIR=\$SLURM_TMPDIR/matplotlib
mkdir -p \$MPLCONFIGDIR

# Set memory and CPU optimizations for Trillium
export OMP_NUM_THREADS=192
export OPENBLAS_NUM_THREADS=192
export MKL_NUM_THREADS=192
export NUMEXPR_NUM_THREADS=192

echo "SLURM job started at \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Virtual environment: \$SCRATCH_ENV"
echo "Running $mol with parameters: $mol_file $n_qubits $n_electrons $pool_type $shots $accuracy $radius $target_accuracy"
echo "Run number: $run/1"

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
LOG_FILE="${script_name}_${mol}_${pool_type}_run${run}_\${SLURM_JOB_ID}.log"

echo "Starting ADAPT-VQE optimization for $mol (run $run/1)..."
echo "Working directory: \$(pwd)"

# Run the ADAPT-VQE script with the exact parameters from notes.txt
python -u $script_name "$mol_file" "$mol" "$n_qubits" "$n_electrons" "$pool_type" "$shots" "$accuracy" "$radius" "$target_accuracy" 2>&1 | tee -a "\$LOG_FILE"

# Check exit status
if [ \$? -eq 0 ]; then
    echo "ADAPT-VQE for $mol (run $run/1) completed successfully at \$(date)"
else
    echo "ADAPT-VQE for $mol (run $run/1) failed with exit code \$? at \$(date)"
fi

# Print final memory usage
echo "Final memory usage:"
free -h

# Copy results to a timestamped directory
TIMESTAMP=\$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results_${script_name}_${mol}_${pool_type}_run${run}_\${TIMESTAMP}_\${SLURM_JOB_ID}"
mkdir -p \$RESULTS_DIR

# Copy output files
echo "Copying results for $mol with $pool_type pool (run $run/1)..."
cp *${mol}*${pool_type}*.csv \$RESULTS_DIR/ 2>/dev/null || echo "No CSV files found for $mol-$pool_type"
cp *${mol}*.json \$RESULTS_DIR/ 2>/dev/null || echo "No JSON cache files found for $mol"
cp "\$LOG_FILE" \$RESULTS_DIR/ 2>/dev/null || echo "No log file found"

# Also copy any generic output files
cp *.csv \$RESULTS_DIR/ 2>/dev/null || echo "No additional CSV files in root directory"
cp *.json \$RESULTS_DIR/ 2>/dev/null || echo "No additional JSON cache files in root directory"

echo "Results copied to: \$RESULTS_DIR"
echo "Job finished at \$(date)"
EOF

        # Submit this individual job
        JOB_ID=$(sbatch temp_job_${mol}_${pool_type}_run${run}.sh | awk '{print $4}')
        echo "    → Submitted as job ID: $JOB_ID"

        # Clean up temporary script
        rm temp_job_${mol}_${pool_type}_run${run}.sh

        # Small delay to avoid overwhelming the scheduler
        sleep 1
    done

    echo "  Completed submitting 5 jobs for $mol with $pool_type pool"
    echo ""

done < notes.txt

echo ""
echo "All jobs submitted successfully!"
echo "Total jobs submitted: $total_jobs"
echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo ""
echo "To cancel all your jobs:"
echo "  scancel -u \$USER"
echo ""
echo "To monitor specific molecule jobs:"
echo "  squeue -u \$USER | grep h4"
echo "  squeue -u \$USER | grep lih"
echo "  squeue -u \$USER | grep beh2"

