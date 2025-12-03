#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "vector.h"
#include "config.h"

// Device pointers declared in vector.h
vector3 *hVel, *d_hVel;
vector3 *hPos, *d_hPos;
double *mass, *d_mass;

// CUDA kernel to compute pairwise accelerations
// Each thread computes the acceleration on entity i due to entity j
__global__ void computeAccelerations(vector3 *d_accels, vector3 *d_pos, double *d_mass, int numentities) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= numentities || j >= numentities) return;
    
    vector3 accel = {0.0, 0.0, 0.0};
    
    if (i != j) {
        vector3 distance;
        distance[0] = d_pos[i][0] - d_pos[j][0];
        distance[1] = d_pos[i][1] - d_pos[j][1];
        distance[2] = d_pos[i][2] - d_pos[j][2];
        
        double magnitude_sq = distance[0]*distance[0] + distance[1]*distance[1] + distance[2]*distance[2];
        double magnitude = sqrt(magnitude_sq);
        double accelmag = -1.0 * GRAV_CONSTANT * d_mass[j] / magnitude_sq;
        
        accel[0] = accelmag * distance[0] / magnitude;
        accel[1] = accelmag * distance[1] / magnitude;
        accel[2] = accelmag * distance[2] / magnitude;
    }
    
    // Store acceleration at [i][j]
    int idx = i * numentities + j;
    d_accels[idx][0] = accel[0];
    d_accels[idx][1] = accel[1];
    d_accels[idx][2] = accel[2];
}

// CUDA kernel to sum accelerations and update velocity/position
// Each thread handles one entity
__global__ void updateVelocityPosition(vector3 *d_accels, vector3 *d_vel, vector3 *d_pos, int numentities) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= numentities) return;
    
    // Sum up all accelerations for entity i
    vector3 accel_sum = {0.0, 0.0, 0.0};
    for (int j = 0; j < numentities; j++) {
        int idx = i * numentities + j;
        accel_sum[0] += d_accels[idx][0];
        accel_sum[1] += d_accels[idx][1];
        accel_sum[2] += d_accels[idx][2];
    }
    
    // Update velocity: v = v + a*t
    d_vel[i][0] += accel_sum[0] * INTERVAL;
    d_vel[i][1] += accel_sum[1] * INTERVAL;
    d_vel[i][2] += accel_sum[2] * INTERVAL;
    
    // Update position: p = p + v*t
    d_pos[i][0] += d_vel[i][0] * INTERVAL;
    d_pos[i][1] += d_vel[i][1] * INTERVAL;
    d_pos[i][2] += d_vel[i][2] * INTERVAL;
}

// Initialize device memory and copy data from host
void initGPU() {
    cudaMalloc((void**)&d_hPos, sizeof(vector3) * NUMENTITIES);
    cudaMalloc((void**)&d_hVel, sizeof(vector3) * NUMENTITIES);
    cudaMalloc((void**)&d_mass, sizeof(double) * NUMENTITIES);
    
    cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);
}

// Copy results back from device to host
void copyFromGPU() {
    cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
}

// Cleanup GPU memory
void cleanupGPU() {
    cudaFree(d_hPos);
    cudaFree(d_hVel);
    cudaFree(d_mass);
}

// Main compute function: Updates the positions and velocities of objects based on gravity
void compute() {
    // Allocate device memory for acceleration matrix
    vector3 *d_accels;
    cudaMalloc((void**)&d_accels, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
    
    // Configure grid and block dimensions
    // For acceleration computation: 2D grid (i, j pairs)
    dim3 blockDim(16, 16, 1);  // 16x16 = 256 threads per block
    dim3 gridDim((NUMENTITIES + blockDim.x - 1) / blockDim.x,
                 (NUMENTITIES + blockDim.y - 1) / blockDim.y,
                 1);
    
    // Compute pairwise accelerations in parallel
    computeAccelerations<<<gridDim, blockDim>>>(d_accels, d_hPos, d_mass, NUMENTITIES);
    cudaDeviceSynchronize();
    
    // Configure 1D grid for velocity/position updates
    int blockSize = 256;
    int gridSize = (NUMENTITIES + blockSize - 1) / blockSize;
    
    // Update velocity and position in parallel
    updateVelocityPosition<<<gridSize, blockSize>>>(d_accels, d_hVel, d_hPos, NUMENTITIES);
    cudaDeviceSynchronize();
    
    // Cleanup acceleration matrix
    cudaFree(d_accels);
}
