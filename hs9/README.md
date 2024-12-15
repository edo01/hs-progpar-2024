# Morphological Operations with CUDA

This project implements **morphological operations** (Erosion and Dilation) in CUDA.
These operations are fundamental in image processing and are commonly used in applications such as object detection, noise reduction, and image preprocessing.

---

## Table of Contents
1. [What are Morphological Operations?](#what-are-morphological-operations)
   - [Erosion](#erosion)
   - [Dilation](#dilation)
2. [How the CUDA Implementation Works](#how-the-cuda-implementation-works)
3. [Building and Running the Code](#building-and-running-the-code)
4. [References](#references)

---

## What are Morphological Operations?

Morphological operations process images based on their shapes, applying a structuring element mask to an input image.
These operations are widely used in binary and grayscale images.

### Erosion

Erosion "shrinks" objects in an image. It removes pixels on the boundaries of objects, reducing the size of foreground regions and increasing the size of background regions.

- **How It Works**: The structuring element mask slides over the image. A pixel in the output image is set to the **minimum** value under the entire mask.
- **Applications**:
  - Remove small noise.
  - Separate objects that are touching.

### Dilation

Dilation "grows" objects in an image. It adds pixels to the boundaries of objects, increasing the size of foreground regions and reducing the size of background regions.

- **How It Works**: The structuring element mask slides over the image. A pixel in the output image is set to the **maximum** value under the entire mask.
- **Applications**:
  - Fill small holes.
  - Connect broken parts of an object.

---

## How the CUDA Implementation Works

This implementation uses **CUDA** to process each pixel in parallel.

1. **Input Image**: A grayscale PGM image (2D array of unsigned 8-bit integers).
2. **Structuring Element**: A small matrix (e.g., 3x3) that defines the neighborhood for morphological operations. This matrix can be modified to change the shape of the structuring element (cross-shaped for instance:  `int h_Mask[] = {0, 1, 0, 1, 1, 1, 0, 1, 0};`)
3. **GPU Kernels**:
   - **Erosion Kernel**: Computes the minimum value in the neighborhood.
   - **Dilation Kernel**: Computes the maximum value in the neighborhood.
4. **Output Image**: The processed image, where each pixel has been updated based on the selected operation.

---

## Building and Running the Code

### Prerequisites
- NVIDIA GPU with CUDA support.
- CUDA Toolkit installed.
- C compiler (e.g., `gcc`).
- Makefile toolchain.

### Steps
1. Get and go to the repository root:
```bash
git clone https://github.com/your-repo/cuda-morphology.git
cd cuda-morphology
```

2. Edit the `Makefile` with your GPU compute capability (here 7.5):
```bash
NVCC_FLAGS = -O3 --gpu-architecture=sm_75 -lineinfo --Werror all-warnings
```

3. Compile the code using `make`
```bash
make
```

4. Run the executable:
```bash
./bin/morpho
```

### Input and Output
- **Input**: A sample grayscale `pgm` image stack `temple_3`
- **Output**: The processed image stack  `pgm` image stack in `output_images`

## References
- [Mathematical_morphology](https://en.wikipedia.org/wiki/Mathematical_morphology)
- [PGM format](https://netpbm.sourceforge.net/doc/pgm.html)
- [MPI-Sintel dataset](http://sintel.is.tue.mpg.de/)