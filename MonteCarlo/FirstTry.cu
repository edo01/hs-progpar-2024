#include <curand_kernel.h>
#include <cmath>
#include <iostream>

const float K = 1.0f;     
const float S0 = 1.0f;    // the spot values
const float v0 = 0.1f;  
const float r = 0.0f;     // the risk-free interest rate
const float kappa = 0.5f; // the mean reversion rate of the volatility
const float theta = 0.1f; // the long-term volatility
const float sigma = 0.3f; // the volatility of volatility
const float rho = -0.7f;  
const int T = 1;          
const int steps = 1000;   
const float dt = 1.0f / steps; 
const int simulations = 100000; 

// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))


__global__ void hestonMonteCarlo(float *d_results, int steps, float dt, float kappa, float theta, float sigma, float rho) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //Initialize the random number generator
    curandState state;
    curand_init(1234, tid, 0, &state);

    float St = S0;
    float vt = v0;

    //Simulation time step
    for (int i = 0; i < steps; ++i) {
        float G1 = curand_normal2(&state);
        float G2 = curand_normal2(&state);

        // Calculate the delta of asset price and volatility
        float dSt = r * St * dt + sqrtf(vt) * St * sqrtf(dt) * (rho * G1 + sqrtf(1 - rho * rho) * G2);
        float dvt = kappa * (theta - vt) * dt + sigma * sqrtf(vt) * sqrtf(dt) * G1;

        St += dSt;
        vt = fabs(vt + dvt); // the function g is either taken to be equal to (·)+ or to | · |
    }
    // E[f(ST )] = E[(S1 − 1)+].
    d_results[tid] = fmaxf(St - K, 0.0f);
}


int main() {
    //Allocate memory on the device to store the results
    float *d_results;
    cudaMalloc((void **)&d_results, simulations * sizeof(float));

    int threadsPerBlock = 256;
    int blocks = (simulations + threadsPerBlock - 1) / threadsPerBlock;

	float Tim;
	cudaEvent_t start, stop;			// GPU timer instructions
	cudaEventCreate(&start);			// GPU timer instructions
	cudaEventCreate(&stop);				// GPU timer instructions
	cudaEventRecord(start, 0);			// GPU timer instructions

    hestonMonteCarlo<<<blocks, threadsPerBlock>>>(d_results, steps, dt, kappa, theta, sigma, rho);

	cudaEventRecord(stop, 0);			// GPU timer instructions
	cudaEventSynchronize(stop);			// GPU timer instructions
	cudaEventElapsedTime(&Tim,			// GPU timer instructions
		start, stop);					// GPU timer instructions
	cudaEventDestroy(start);			// GPU timer instructions
	cudaEventDestroy(stop);				// GPU timer instructions

    float *h_results = (float *)malloc(simulations * sizeof(float));
    cudaMemcpy(h_results, d_results, simulations * sizeof(float), cudaMemcpyDeviceToHost);

    float option_price = 0.0f;
    for (int i = 0; i < simulations; ++i) {
        option_price += h_results[i];
    }
    option_price /= simulations;
    option_price *= expf(-r * T); 

    std::cout << "Option Price: " << option_price << std::endl;

    free(h_results);
    cudaFree(d_results);

    return 0;
}
