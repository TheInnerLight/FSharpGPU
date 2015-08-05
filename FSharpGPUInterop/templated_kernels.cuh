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

/* This software contains source code provided by NVIDIA Corporation. */

/* Copyright © 2015 Philip Curzon */

#include "definitions.cuh"
#include "functions.cuh"
#include "scankernels.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <algorithm>

#pragma once

template<typename T>
__device__ void getInputArrayValueForIndexingScheme(int pos, T *inputArr, const int inputOffset, const int inputN, int scheme, T *val)
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

// arthimetic functions

template<typename T, typename U> __device__ U _kernel_add(T elem1, T elem2) { return elem1 + elem2; }
template<typename T, typename U> __device__ U _kernel_subtract(T elem1, T elem2) { return elem1 - elem2; }
template<typename T, typename U> __device__ U _kernel_multiply(T elem1, T elem2) { return elem1 * elem2; }
template<typename T, typename U> __device__ U _kernel_divide(T elem1, T elem2) { return elem1 / elem2; }
template<typename T, typename U> __device__ U _kernel_power(T elem1, T elem2) { return pow(elem1, elem2); }

// comparison functions

template<typename T> __device__ __int32 _kernel_greater_than(T elem1, T elem2) { return elem1 > elem2; }
template<typename T> __device__ __int32 _kernel_greater_than_or_equal(T elem1, T elem2) { return elem1 >= elem2; }
template<typename T> __device__ __int32 _kernel_less_than(T elem1, T elem2) { return elem1 < elem2; }
template<typename T> __device__ __int32 _kernel_less_than_or_equal(T elem1, T elem2) { return elem1 <= elem2; }

// equality functions

template<typename T> __device__ __int32 _kernel_equality(T elem1, T elem2) { return elem1 == elem2; }
template<typename T> __device__ __int32 _kernel_inequality(T elem1, T elem2) { return elem1 != elem2; }

// conditional functions

__device__ __int32 _kernel_conditional_and(__int32 elem1, __int32 elem2) { return elem1 && elem2; }
__device__ __int32 _kernel_conditional_or(__int32 elem1, __int32 elem2) { return elem1 || elem2; }

// maths functions

template<typename T, typename U> __device__ U _kernel_sqrt(T elem) { return sqrt(elem); }
template<typename T, typename U> __device__ U _kernel_sin(T elem) { return sin(elem); }
template<typename T, typename U> __device__ U _kernel_cos(T elem) { return cos(elem); }
template<typename T, typename U> __device__ U _kernel_tan(T elem) { return tan(elem); }
template<typename T, typename U> __device__ U _kernel_sinh(T elem) { return sinh(elem); }
template<typename T, typename U> __device__ U _kernel_cosh(T elem) { return cosh(elem); }
template<typename T, typename U> __device__ U _kernel_tanh(T elem) { return tanh(elem); }
template<typename T, typename U> __device__ U _kernel_arcsin(T elem) { return asin(elem); }
template<typename T, typename U> __device__ U _kernel_arccos(T elem) { return acos(elem); }
template<typename T, typename U> __device__ U _kernel_arctan(T elem) { return atan(elem); }
template<typename T, typename U> __device__ U _kernel_log(T elem) { return log(elem); }
template<typename T, typename U> __device__ U _kernel_log10(T elem) { return log10(elem); }
template<typename T, typename U> __device__ U _kernel_exp(T elem) { return exp(elem); }


template<typename T, typename U>
__global__ void _kernel_map_op(T *inputArr, const int inputOffset, const ThreadBlocks inputN, U *outputArr, U p_function(T))
{
	T val;
	for (int i = 0; i < inputN.loopCount; ++i)
	{
		getInputArrayValueForIndexingScheme<T>(i*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x, inputArr, inputOffset, inputN.N, 0, &val);
		outputArr[i*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x] = p_function(val);
	}
}

template<typename T, typename U>
__global__ void _kernel_map_with_const_op(T *inputArr, const int inputOffset, const ThreadBlocks inputN, const T d, U *outputArr, U p_function(T, T))
{
	T val;
	for (int i = 0; i < inputN.loopCount; ++i)
	{
		getInputArrayValueForIndexingScheme<T>(i*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x, inputArr, inputOffset, inputN.N, 0, &val);
		outputArr[i*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x] = p_function(val, d);
	}
}

template<typename T, typename U>
__global__ void _kernel_map_with_const_op2(T *inputArr, const int inputOffset, const ThreadBlocks inputN, const T d, U *outputArr, U p_function(T, T))
{
	T val;
	for (int i = 0; i < inputN.loopCount; ++i)
	{
		getInputArrayValueForIndexingScheme<T>(i*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x, inputArr, inputOffset, inputN.N, 0, &val);
		outputArr[i*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x] = p_function(d, val);
	}
}

template<typename T, typename U>
__global__ void _kernel_map2_op(T *input1Arr, const int input1Offset, T *input2Arr, const int input2Offset, const ThreadBlocks inputN, U *outputArr, U p_function(T, T))
{
	T val1, val2;
	for (int i = 0; i < inputN.loopCount; ++i)
	{
		getInputArrayValueForIndexingScheme<T>(i*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x, input1Arr, input1Offset, inputN.N, 0, &val1);
		getInputArrayValueForIndexingScheme<T>(i*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x, input2Arr, input2Offset, inputN.N, 0, &val2);
		T newVal = p_function(val1, val2);
		outputArr[i*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x] = p_function(val1, val2);
	}
}

/* Reduce to half the size */
template<typename T>
__global__ void _kernel_reduce_to_half(T *inputArr, const int inputOffset, const ThreadBlocks inputN, T *outputArr)
{
	T val;
	for (int i = 0; i < inputN.loopCount; ++i)
	{
		getInputArrayValueForIndexingScheme(i*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x, inputArr, inputOffset, inputN.N, 0, &val);
		if ((i*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x) % 2 == 0)
			outputArr[(i*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x) / 2] = val;
	}
}