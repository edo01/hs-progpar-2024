# hs-progpar-2024
Hands-on session from the course "Programmation parallèle pour systèmes embarqués - S1-24"

# README.md

## Summary

This repository contains a series of optimizations for a blur kernel applied to images. The primary goal of this project is to improve the performance of the blur operation by progressively optimizing the original implementation. The optimizations include techniques such as loop unrolling, variable rotation, function inlining, and optimized handling of image boundaries. Each optimization is tested with different compiler optimization levels to measure its effect on execution time. The key files in this project are the report detailing the optimizations, the `blur.c` and `blur_v2.c` source files, and two script (`run.sh` and `test.sh`) to automate performance testing.

---

## File Descriptions

### 1. **Report**
This file provides a detailed explanation of the optimizations applied to the blur kernel. It discusses each step of the optimization process, starting from the default implementation and progressing through various techniques like loop unrolling, function inlining, and border handling optimization. The report includes performance analysis based on execution time measurements and how the optimizations impact both the visual quality of the output and computational efficiency.

### 2. **blur.c**
This file contains the base implementation of the blur kernel, where multiple versions of the `blur_do_tile_default` function are implemented. Each version is a variant of the original, where different optimization techniques are applied. The main focus of this file is exploring different ways to reduce computational overhead and improve performance.

### 3. **blur_v2.c**
This file includes the second version of the blur kernel, where further optimizations have been introduced, especially in the management of image borders. The improvements primarily focus on coalescing memory accesses for edge pixels, minimizing redundant computations, and ensuring that the blur effect is applied efficiently without degrading the quality of border pixel processing. This file builds on the optimizations introduced in `blur.c`.

### 4. **run.sh**
This script automates the process of running the various kernel optimization implementations. It executes each version of the blur kernel multiple times to obtain more accurate execution time measurements, mitigating the impact of system load variations. The script also compiles the program using different optimization flags (`-O0`, `-O1`, `-O2`, `-O3`) to assess how different levels of compiler optimization affect performance. This allows for a thorough comparison of the optimizations applied at both the code and compiler levels.

### 5. **test.sh**
The test.sh script is used for testing different kernel optimization implementations and validating their correctness by comparing the results to predefined hash values. It runs the blur kernel on the CPU using taskset to bind the process to a specific core (core 1), ensuring consistent performance measurements. For each kernel variant (e.g., default, default_nb, optim1, optim2, etc.), the script executes the kernel and compares the output against a reference hash to verify correctness.

---

## Usage

To compile and run the program with different optimization levels:
1. Make the script executable:
   ```bash
   chmod +x run.sh
   ```
2. Run the script:
   ```bash
   ./run.sh
   ```
or 
   ```bash
   ./run.sh -c
   ```

This will execute each version of the kernel and generate performance data for analysis. The script is configured to run each kernel multiple times to ensure the accuracy of the timing results.



