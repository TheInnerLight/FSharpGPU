#include "definitions.cuh"
#include "functions.cuh"
#include "scankernels.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <algorithm>

#pragma once

template<typename T>
__device__ void getInputArrayValueForIndexingScheme1000(int pos, T *inputArr, const int inputOffset, const int inputN, int scheme, T *val)
{
	//printf("Array value function called\n");
	switch (scheme)
	{
	case 0:
		if (((blockIdx.x * blockDim.x + threadIdx.x + inputOffset) >= inputN) || ((blockIdx.x * blockDim.x + threadIdx.x + inputOffset) < 0)) *val = 0.0;
		else *val = inputArr[blockIdx.x * blockDim.x + threadIdx.x + inputOffset];
		break;
	default:
		*val = inputArr[blockIdx.x * blockDim.x + threadIdx.x + inputOffset % inputN];
	}
}

template<typename T, typename U>
__device__ U _kernel_add(T elem1, T elem2)
{
	return elem1 + elem2;
}

template<typename T, typename U>
__device__ U _kernel_subtract(T elem1, T elem2)
{
	return elem1 - elem2;
}

template<typename T, typename U>
__device__ U _kernel_multiply(T elem1, T elem2)
{
	return elem1 * elem2;
}

template<typename T, typename U>
__device__ U _kernel_divide(T elem1, T elem2)
{
	return elem1 / elem2;
}

template<typename T, typename U>
__device__ U _kernel_power(T elem1, T elem2)
{
	return pow(elem1, elem2);
}

template<typename T, typename U>
__global__ void _kernel_map_op(T *inputArr, const int inputOffset, const ThreadBlocks inputN, const T d, U *outputArr, U p_function(T, T))
{
	T val;
	for (int i = 0; i < inputN.loopCount; ++i)
	{
		getInputArrayValueForIndexingScheme1000<T>(i*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x, inputArr, inputOffset, inputN.N, 0, &val);
		U newVal = p_function(val, d);
		outputArr[i*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x] = p_function(val, d);
	}
}

template<typename T, typename U>
__global__ void _kernel_map_op2(T *inputArr, const int inputOffset, const ThreadBlocks inputN, const T d, T *outputArr, U p_function(T, T))
{
	T val;
	for (int i = 0; i < inputN.loopCount; ++i)
	{
		getInputArrayValueForIndexingScheme1000<T>(i*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x, inputArr, inputOffset, inputN.N, 0, &val);
		outputArr[i*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x] = p_function(d, val);
	}
}

template<typename T, typename U>
__global__ void _kernel_map2_op(T *input1Arr, const int input1Offset, T *input2Arr, const int input2Offset, const ThreadBlocks inputN, U *outputArr, U p_function(T, T))
{
	T val1, val2;
	for (int i = 0; i < inputN.loopCount; ++i)
	{
		getInputArrayValueForIndexingScheme1000<T>(i*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x, input1Arr, input1Offset, inputN.N, 0, &val1);
		getInputArrayValueForIndexingScheme1000<T>(i*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x, input2Arr, input2Offset, inputN.N, 0, &val2);
		T newVal = p_function(val1, val2);
		outputArr[i*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x] = p_function(val1, val2);
	}
}