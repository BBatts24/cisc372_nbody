// compute.h
// compute.cu
// Compile with: nvcc -O2 -std=c++11 -arch=sm_60 -o nbody compute.cu nbody.c ...
// (adjust arch as appropriate)

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "compute.h"

// small softening to avoid divide-by-zero / singularities
#define SOFTENING 1e-9

// simple CUDA error check
#define CUDA_CALL(call)                                                                             \
    do                                                                                              \
    {                                                                                               \
        cudaError_t err = (call);                                                                   \
        if (err != cudaSuccess)                                                                     \
        {                                                                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                     \
        }                                                                                           \
    } while (0)

// Device pointers (kept static within this compilation unit)
static double *dPosX = NULL, *dPosY = NULL, *dPosZ = NULL;
static double *dVelX = NULL, *dVelY = NULL, *dVelZ = NULL;
static double *dMass = NULL;
static double *dAccelX = NULL, *dAccelY = NULL, *dAccelZ = NULL;

// Kernels

// each thread computes the net acceleration for body i by summing over j
__global__ void kernel_compute_accel(int N,
                                     const double *posX, const double *posY, const double *posZ,
                                     const double *mass,
                                     double *outAx, double *outAy, double *outAz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    double xi = posX[i];
    double yi = posY[i];
    double zi = posZ[i];

    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;

    // naive O(N) reduction per thread
    for (int j = 0; j < N; ++j)
    {
        if (j == i)
            continue;
        double dx = xi - posX[j];
        double dy = yi - posY[j];
        double dz = zi - posZ[j];
        double dist2 = dx * dx + dy * dy + dz * dz + SOFTENING;
        double invDist = 1.0 / sqrt(dist2);            // 1/sqrt(dist2)
        double invDist3 = invDist * invDist * invDist; // 1/r^3
        // acceleration contribution: -G * m_j * (r_vec) / r^3
        double factor = -(double)GRAV_CONSTANT * mass[j] * invDist3;
        ax += factor * dx;
        ay += factor * dy;
        az += factor * dz;
    }

    outAx[i] = ax;
    outAy[i] = ay;
    outAz[i] = az;
}

// update velocities and positions
__global__ void kernel_update(int N,
                              double *velX, double *velY, double *velZ,
                              double *posX, double *posY, double *posZ,
                              const double *accelX, const double *accelY, const double *accelZ,
                              double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    double vx = velX[i] + accelX[i] * dt;
    double vy = velY[i] + accelY[i] * dt;
    double vz = velZ[i] + accelZ[i] * dt;

    double px = posX[i] + vx * dt;
    double py = posY[i] + vy * dt;
    double pz = posZ[i] + vz * dt;

    velX[i] = vx;
    velY[i] = vy;
    velZ[i] = vz;

    posX[i] = px;
    posY[i] = py;
    posZ[i] = pz;
}

// Free GPU allocations (called from freeHostMemory in nbody.c)
void gpuFree()
{
    if (dPosX)
    {
        cudaFree(dPosX);
        dPosX = NULL;
    }
    if (dPosY)
    {
        cudaFree(dPosY);
        dPosY = NULL;
    }
    if (dPosZ)
    {
        cudaFree(dPosZ);
        dPosZ = NULL;
    }
    if (dVelX)
    {
        cudaFree(dVelX);
        dVelX = NULL;
    }
    if (dVelY)
    {
        cudaFree(dVelY);
        dVelY = NULL;
    }
    if (dVelZ)
    {
        cudaFree(dVelZ);
        dVelZ = NULL;
    }
    if (dMass)
    {
        cudaFree(dMass);
        dMass = NULL;
    }
    if (dAccelX)
    {
        cudaFree(dAccelX);
        dAccelX = NULL;
    }
    if (dAccelY)
    {
        cudaFree(dAccelY);
        dAccelY = NULL;
    }
    if (dAccelZ)
    {
        cudaFree(dAccelZ);
        dAccelZ = NULL;
    }
}

// Public compute() function
// Note: hPos, hVel, mass are declared in nbody.c as extern globals.
// We'll refer to them here as externs. They must be defined in the final link unit.
extern vector3 *hPos;
extern vector3 *hVel;
extern double *mass;

void compute()
{
    const int N = NUMENTITIES;
    if (N <= 0)
        return;

    // create host interleaved arrays for copying to device
    double *hPosX = (double *)malloc(sizeof(double) * N);
    double *hPosY = (double *)malloc(sizeof(double) * N);
    double *hPosZ = (double *)malloc(sizeof(double) * N);
    double *hVelX = (double *)malloc(sizeof(double) * N);
    double *hVelY = (double *)malloc(sizeof(double) * N);
    double *hVelZ = (double *)malloc(sizeof(double) * N);
    double *hMass = (double *)malloc(sizeof(double) * N);

    if (!hPosX || !hPosY || !hPosZ || !hVelX || !hVelY || !hVelZ || !hMass)
    {
        fprintf(stderr, "Host temporary allocation failed in compute()\n");
        exit(EXIT_FAILURE);
    }

    // copy from vector3 arrays to contiguous arrays
    for (int i = 0; i < N; ++i)
    {
        hPosX[i] = hPos[i][0];
        hPosY[i] = hPos[i][1];
        hPosZ[i] = hPos[i][2];
        hVelX[i] = hVel[i][0];
        hVelY[i] = hVel[i][1];
        hVelZ[i] = hVel[i][2];
        hMass[i] = mass[i];
    }

    // allocate device memory on first call
    if (!dPosX)
    {
        CUDA_CALL(cudaMalloc((void **)&dPosX, sizeof(double) * N));
        CUDA_CALL(cudaMalloc((void **)&dPosY, sizeof(double) * N));
        CUDA_CALL(cudaMalloc((void **)&dPosZ, sizeof(double) * N));
        CUDA_CALL(cudaMalloc((void **)&dVelX, sizeof(double) * N));
        CUDA_CALL(cudaMalloc((void **)&dVelY, sizeof(double) * N));
        CUDA_CALL(cudaMalloc((void **)&dVelZ, sizeof(double) * N));
        CUDA_CALL(cudaMalloc((void **)&dMass, sizeof(double) * N));
        CUDA_CALL(cudaMalloc((void **)&dAccelX, sizeof(double) * N));
        CUDA_CALL(cudaMalloc((void **)&dAccelY, sizeof(double) * N));
        CUDA_CALL(cudaMalloc((void **)&dAccelZ, sizeof(double) * N));
    }

    // copy host -> device
    CUDA_CALL(cudaMemcpy(dPosX, hPosX, sizeof(double) * N, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dPosY, hPosY, sizeof(double) * N, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dPosZ, hPosZ, sizeof(double) * N, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dVelX, hVelX, sizeof(double) * N, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dVelY, hVelY, sizeof(double) * N, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dVelZ, hVelZ, sizeof(double) * N, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dMass, hMass, sizeof(double) * N, cudaMemcpyHostToDevice));

    // kernel launch params
    const int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // compute accelerations
    kernel_compute_accel<<<blocks, threadsPerBlock>>>(N, dPosX, dPosY, dPosZ, dMass, dAccelX, dAccelY, dAccelZ);
    CUDA_CALL(cudaGetLastError());

    // update velocities and positions
    double dt = INTERVAL;
    kernel_update<<<blocks, threadsPerBlock>>>(N, dVelX, dVelY, dVelZ, dPosX, dPosY, dPosZ, dAccelX, dAccelY, dAccelZ, dt);
    CUDA_CALL(cudaGetLastError());

    // copy device -> host (updated positions & velocities)
    CUDA_CALL(cudaMemcpy(hPosX, dPosX, sizeof(double) * N, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(hPosY, dPosY, sizeof(double) * N, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(hPosZ, dPosZ, sizeof(double) * N, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(hVelX, dVelX, sizeof(double) * N, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(hVelY, dVelY, sizeof(double) * N, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(hVelZ, dVelZ, sizeof(double) * N, cudaMemcpyDeviceToHost));

    // write back into original host vector3 arrays
    for (int i = 0; i < N; ++i)
    {
        hPos[i][0] = hPosX[i];
        hPos[i][1] = hPosY[i];
        hPos[i][2] = hPosZ[i];
        hVel[i][0] = hVelX[i];
        hVel[i][1] = hVelY[i];
        hVel[i][2] = hVelZ[i];
    }

    // free temporaries
    free(hPosX);
    free(hPosY);
    free(hPosZ);
    free(hVelX);
    free(hVelY);
    free(hVelZ);
    free(hMass);

    // keep device allocations for reuse across compute() calls
}