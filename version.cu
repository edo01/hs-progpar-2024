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

        // Update asset price and volatility, ensuring volatility is non-negative
        St += dSt;
        vt = fabs(vt + dvt); 
    }

    // (S_T - K)^+
    d_results[tid] = fmaxf(St - K, 0.0f);
}


int main() {
    //Allocate memory on the device to store the results
    float *d_results;
    cudaMalloc((void **)&d_results, simulations * sizeof(float));

    int threadsPerBlock = 256;
    int blocks = (simulations + threadsPerBlock - 1) / threadsPerBlock;

    hestonMonteCarlo<<<blocks, threadsPerBlock>>>(d_results, steps, dt, kappa, theta, sigma, rho);

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
