#!/bin/bash

# Script to submit multiple individual ADAPT-VQE jobs
# Each job will be completely separate with its own job ID

echo "Submitting individual ADAPT-VQE jobs..."

# Define the same parameter combinations from execution.sh
declare -a JOB_PARAMS=(
    "lih_fer.bin lih 12 4 uccsd"
    "beh2_fer.bin beh2 14 4 uccsd"
)

# Submit each job individually
for i in "${!JOB_PARAMS[@]}"; do
    # Parse parameters
    read -r mol_file mol n_qubits n_electrons pool_type <<< "${JOB_PARAMS[$i]}"

    echo "Submitting job $((i+1))/6: $mol with $pool_type pool"

    # Create temporary individual job script
    cat > temp_job_${i}.sh << EOF
#!/bin/bash
#SBATCH --account=rrg-izmaylov
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=23:30:00
#SBATCH --job-name=adapt_${mol}_${pool_type}
#SBATCH --output=adapt_${mol}_${pool_type}_%j.out
#SBATCH --error=adapt_${mol}_${pool_type}_%j.err
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
echo "Running $mol with parameters: $mol_file $n_qubits $n_electrons $pool_type"

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
LOG_FILE="adapt_vqe_${mol}_${pool_type}_\${SLURM_JOB_ID}.log"

echo "Starting ADAPT-VQE qubitwise optimization for $mol..."
echo "Working directory: \$(pwd)"

# Run the ADAPT-VQE script
python -u adapt_vqe_qiskit_qubitwise_bai.py "$mol_file" "$mol" "$n_qubits" "$n_electrons" "$pool_type" 2>&1 | tee -a "\$LOG_FILE"

# Check exit status
if [ \$? -eq 0 ]; then
    echo "ADAPT-VQE for $mol completed successfully at \$(date)"
else
    echo "ADAPT-VQE for $mol failed with exit code \$? at \$(date)"
fi

# Print final memory usage
echo "Final memory usage:"
free -h

# Copy results to a timestamped directory
TIMESTAMP=\$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results_${mol}_${pool_type}_\${TIMESTAMP}_\${SLURM_JOB_ID}"
mkdir -p \$RESULTS_DIR

# Copy output files
echo "Copying results for $mol with $pool_type pool..."
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
    JOB_ID=$(sbatch temp_job_${i}.sh | awk '{print $4}')
    echo "  → Submitted as job ID: $JOB_ID"

    # Clean up temporary script
    rm temp_job_${i}.sh

    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

echo ""
echo "All 6 jobs submitted successfully!"
echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo ""
echo "To cancel all your jobs:"
echo "  scancel -u \$USER"
