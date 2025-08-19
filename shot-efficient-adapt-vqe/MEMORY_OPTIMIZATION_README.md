# Memory Optimization for ADAPT-VQE

This directory contains tools to analyze and optimize memory usage for ADAPT-VQE simulations without modifying the main algorithms.

## Problem

The original ADAPT-VQE code can run out of memory on local computers when simulating larger systems like LiH, even though the research paper was able to get results (likely using high-performance computing clusters).

## Solutions

### 1. Memory Monitoring Wrapper (`run_with_memory_monitoring.py`)

A simple wrapper that runs the original `example_lih.py` with memory monitoring:

```bash
python run_with_memory_monitoring.py
```

This will:
- Monitor memory usage throughout execution
- Report peak memory usage
- Provide recommendations if memory issues occur

### 2. Memory-Optimized Example (`example_lih_memory_optimized.py`)

A version of the original example with reduced parameters to prevent memory issues:

```bash
python example_lih_memory_optimized.py
```

**Key changes:**
- `max_adapt_iter`: 100 → 30
- `max_opt_iter`: 100 → 30  
- `k`: 100 → 50
- `shots_budget`: 1024 → 512
- `N_experiments`: 2 → 1

### 3. Advanced Memory Optimizer (`memory_optimized_runner.py`)

A comprehensive tool with command-line interface for running ADAPT-VQE with memory optimization:

```bash
# Install dependencies
pip install -r requirements_memory_monitoring.txt

# Run with default settings
python memory_optimized_runner.py --molecule lih --pool SD

# Run with memory limit
python memory_optimized_runner.py --molecule lih --pool SD --max-memory 4000

# Run with custom parameters
python memory_optimized_runner.py --molecule lih --pool SD --max-adapt-iter 20 --max-opt-iter 20 --shots-budget 256
```

## Memory Optimization Strategies

### 1. Parameter Reduction
- Reduce `max_adapt_iter` and `max_opt_iter`
- Reduce `shots_budget` and `k`
- Set `N_experiments=1`

### 2. Environment Optimization
- Set `OMP_NUM_THREADS=1`
- Set `OPENBLAS_NUM_THREADS=1`
- Set `MKL_NUM_THREADS=1`

### 3. Garbage Collection
- Force garbage collection between stages
- Monitor memory usage and clean up when needed

### 4. Lazy Loading
- Import modules only when needed
- Use context managers for resource management

## Usage Examples

### For Local Development
```bash
# Start with memory-optimized version
python example_lih_memory_optimized.py

# If successful, gradually increase parameters
# Edit example_lih_memory_optimized.py and increase values
```

### For Cluster Submission
```bash
# Use the original submit_adapt_vqe_jobs.sh
# It will create modified versions with different pool types
./submit_adapt_vqe_jobs.sh
```

### For Memory Analysis
```bash
# Analyze memory usage of original code
python run_with_memory_monitoring.py

# Use advanced optimizer with analysis
python memory_optimized_runner.py --molecule lih --pool SD
```

## Troubleshooting

### Memory Errors
If you encounter memory errors:

1. **Reduce parameters further:**
   ```python
   max_adapt_iter=20  # Even smaller
   max_opt_iter=20    # Even smaller
   shots_budget=256   # Even smaller
   ```

2. **Use smaller molecules:**
   - Start with H2 instead of LiH
   - Use smaller basis sets

3. **Use smaller operator pools:**
   - Try QE instead of SD
   - Use qubit_pool for smaller systems

### Performance vs Memory Trade-offs
- **Lower memory usage** = Fewer iterations, fewer shots, less accuracy
- **Higher accuracy** = More memory usage, longer runtime
- **Balance** = Start with memory-optimized settings and gradually increase

## Expected Memory Usage

| System | Pool Type | Expected Memory (MB) | Recommended Settings |
|--------|-----------|---------------------|---------------------|
| H2 | SD | 100-500 | Default |
| H2 | QE | 50-200 | Default |
| LiH | SD | 1000-4000 | Memory-optimized |
| LiH | QE | 500-2000 | Memory-optimized |
| BeH2 | SD | 2000-8000 | Very conservative |

## Cluster vs Local

The research paper likely used:
- **High-memory nodes** (32GB+ RAM)
- **Multiple cores** for parallel processing
- **Longer runtimes** (hours vs minutes)
- **Larger parameter values** (max_adapt_iter=100+)

For local development, use the memory-optimized versions to get similar results with lower resource requirements. 