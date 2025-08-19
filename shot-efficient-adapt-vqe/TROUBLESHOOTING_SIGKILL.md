# Troubleshooting SIGKILL Memory Issues

## 🚨 Problem: SIGKILL (Signal 9)
Your ADAPT-VQE process is being killed by the system due to excessive memory usage.

## 🔍 Root Cause Analysis

The memory explosion (246MB → 400GB+) suggests:

1. **Exponential memory scaling** with system size
2. **Memory leak** in the ADAPT-VQE implementation
3. **Large matrix allocations** during gradient calculations
4. **Inefficient memory management** in the underlying libraries

## 🛠️ Solutions (Try in Order)

### 1. **Test with H2 First** (Recommended)
```bash
python example_h2_minimal.py
```
- H2 is much smaller (2 qubits vs 12 qubits for LiH)
- Will help identify if the issue is specific to LiH or general

### 2. **Use Memory Monitoring**
```bash
# Run with 2GB memory limit
python run_with_memory_limit.py example_lih_minimal.py --max-memory 2000

# Run with 1GB memory limit
python run_with_memory_limit.py example_lih_minimal.py --max-memory 1000
```

### 3. **Step-by-Step Diagnosis**
```bash
python memory_diagnostic.py
```
This will identify exactly where the memory spike occurs.

### 4. **Ultra-Conservative Parameters**
```bash
python example_lih_minimal.py
```
Uses the most minimal parameters possible.

## 📊 Memory Usage Guidelines

| System | Pool Type | Safe Memory Limit | Recommended Settings |
|--------|-----------|------------------|---------------------|
| H2 | SD | 500MB | Default |
| H2 | QE | 300MB | Default |
| LiH | QE | 1000MB | Ultra-conservative |
| LiH | SD | 2000MB | Very conservative |
| BeH2 | Any | 4000MB+ | Cluster only |

## 🔧 Parameter Optimization

### For LiH with Limited Memory:

**Ultra-Conservative:**
```python
max_adapt_iter=1
max_opt_iter=1
shots_budget=32
k=5
N_experiments=1
```

**Conservative:**
```python
max_adapt_iter=5
max_opt_iter=5
shots_budget=64
k=10
N_experiments=1
```

**Moderate:**
```python
max_adapt_iter=10
max_opt_iter=10
shots_budget=128
k=20
N_experiments=1
```

## 🎯 Progressive Testing Strategy

### Step 1: Verify Basic Functionality
```bash
python example_h2_minimal.py
```
If this fails, the issue is fundamental to the code.

### Step 2: Test LiH with Minimal Settings
```bash
python example_lih_minimal.py
```
Use the interactive version to stop before running the algorithm.

### Step 3: Gradual Scaling
If minimal settings work, gradually increase:
1. `max_adapt_iter`: 1 → 2 → 5 → 10
2. `shots_budget`: 32 → 64 → 128 → 256
3. `k`: 5 → 10 → 20 → 50

### Step 4: Memory Monitoring
Use the memory monitor to find your system's limits:
```bash
python run_with_memory_limit.py example_lih_minimal.py --max-memory 1500
```

## 🖥️ System-Specific Solutions

### macOS (Your System)
- **Available RAM**: Check with `top` or Activity Monitor
- **Swap Space**: Limited on macOS
- **Solution**: Use more conservative parameters

### Linux
- **Available RAM**: Check with `free -h`
- **Swap Space**: Can be increased
- **Solution**: Add swap space if needed

### Windows
- **Available RAM**: Check Task Manager
- **Virtual Memory**: Can be increased
- **Solution**: Increase page file size

## 🔍 Debugging Commands

### Check Available Memory
```bash
# macOS
top -l 1 | grep PhysMem

# Linux
free -h

# Windows
wmic computersystem get TotalPhysicalMemory
```

### Monitor Memory in Real-Time
```bash
# macOS/Linux
watch -n 1 'ps aux | grep python'

# Or use Activity Monitor (macOS)
open -a "Activity Monitor"
```

### Check Process Memory
```bash
ps aux | grep python
```

## 🚀 Alternative Approaches

### 1. **Use Smaller Basis Sets**
- Try different basis sets for LiH
- Smaller basis = fewer qubits = less memory

### 2. **Use Different Operator Pools**
- QE pool is smaller than SD pool
- Try qubit_pool for even smaller systems

### 3. **Run on Cluster**
- Use the provided `submit_adapt_vqe_jobs.sh`
- Clusters have more memory and better resource management

### 4. **Use Cloud Computing**
- Google Colab (free, limited memory)
- AWS/GCP instances with more RAM

## 📈 Expected Results

### Successful Run Indicators:
- Memory usage stays under 2GB
- Process completes without SIGKILL
- Energy values are reasonable
- No memory errors in output

### Failure Indicators:
- Memory usage spikes rapidly
- Process killed by system
- Memory errors in output
- Virtual memory exceeds physical RAM

## 🆘 When All Else Fails

1. **Use H2 instead of LiH** for development/testing
2. **Run on a machine with more RAM** (16GB+)
3. **Use cloud computing resources**
4. **Contact the original authors** for optimized versions
5. **Consider using a different ADAPT-VQE implementation**

## 📞 Getting Help

If you're still having issues:

1. Run the diagnostic scripts and share the output
2. Check your system's available memory
3. Try the H2 test first
4. Use the memory monitoring tools
5. Consider running on a cluster or cloud platform

The key is to start small (H2) and gradually scale up while monitoring memory usage! 