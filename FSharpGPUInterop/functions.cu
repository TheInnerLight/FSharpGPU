/*This file is part of FSharpGPU.

	FSharpGPU is free software : you can redistribute it and / or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	FSharpGPU is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with FSharpGPU.If not, see <http://www.gnu.org/licenses/>.
*/

/* Copyright © 2015 Philip Curzon */

#include "definitions.cuh"
#include "functions.cuh"


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <functional>
#include <algorithm>

/* Create an uninitialised cuda array of length n, where each element has size typeSize */
int createCUDAArray(int n, int typeSize, void **devPtr)
{
	cudaError_t cudaStatus;
	int byteSize = n * typeSize;
	if ((cudaStatus = cudaMalloc(devPtr, byteSize)) != cudaSuccess) return cudaStatus;
	return cudaStatus;
}

/* Free a cuda array */
int freeCUDAArray(void *devPtr)
{
	return cudaFree(devPtr);
}

/* Create an uninitialised array of doubles of length n */
int createCUDADoubleArray(int n, double **devPtr)
{
	cudaError_t cudaStatus;
	int byteSize = n * sizeof(double);
	if ((cudaStatus = cudaMalloc((void**)devPtr, byteSize)) != cudaSuccess) return cudaStatus;
	return cudaStatus;
}



/* Create and initialise array of doubles of length n */
int initialiseCUDADoubleArray(const double *array, const int n, double **devPtr)
{
	cudaError_t cudaStatus;
	int byteSize = n * sizeof(double);
	if ((cudaStatus = cudaMalloc((void**)devPtr, byteSize)) != cudaSuccess) return cudaStatus;
	if ((cudaStatus = cudaMemcpy(*devPtr, array, byteSize, cudaMemcpyHostToDevice)) != cudaSuccess) return cudaStatus;
	return cudaStatus;
}

/* Retreive the contents of an array of cuda doubles */
int retrieveCUDADoubleArray(double *devPtr, const int offset, double dblArray[], const int n)
{
	cudaError_t cudaStatus;
	int byteSize = n * sizeof(double);
	if ((cudaStatus = cudaMemcpy(dblArray, devPtr+offset, byteSize, cudaMemcpyDeviceToHost)) != cudaSuccess) return cudaStatus;
	return cudaStatus;
}

int createCUDABoolArray(int n, int **devPtr)
{
	cudaError_t cudaStatus;
	int byteSize = n * sizeof(int);
	if ((cudaStatus = cudaMalloc((void**)devPtr, byteSize)) != cudaSuccess) return cudaStatus;
	return cudaStatus;
}

int initialiseCUDABoolArray(const int *array, const int n, int **devPtr)
{
	cudaError_t cudaStatus;
	int byteSize = n * sizeof(bool);
	if ((cudaStatus = cudaMalloc((void**)devPtr, byteSize)) != cudaSuccess) return cudaStatus;
	if ((cudaStatus = cudaMemcpy(*devPtr, array, byteSize, cudaMemcpyHostToDevice)) != cudaSuccess) return cudaStatus;
	return cudaStatus;
}

int retrieveCUDABoolArray(int *devPtr, const int offset, int dblArray[], const int n)
{
	cudaError_t cudaStatus;
	int byteSize = n * sizeof(int);
	if ((cudaStatus = cudaMemcpy(dblArray, devPtr + offset, byteSize, cudaMemcpyDeviceToHost)) != cudaSuccess) return cudaStatus;
	return cudaStatus;
}
