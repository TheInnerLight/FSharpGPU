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

/* Copyright � 2015 Philip Curzon */

#include "kernels.cuh"
#include "functions.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>



/* Split the size of the array between threads and blocks */
ThreadBlocks getThreadsAndBlocks(const int n) 
{
	ThreadBlocks tb;
	tb.threadCount = std::min(512, n);
	tb.blockCount = std::max(1u, (n + tb.threadCount - 1) / tb.threadCount);
	return tb;
}

__device__ void getInputArrayValueForIndexingScheme(double *inputArr, const int inputOffset, const int inputN, int scheme, double *val)
{
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

/******************************************************************************************************************/
/* double to double kernel maps */
/******************************************************************************************************************/

/* Kernel for adding a constant to an array */
__global__ void _kernel_ddmapAddSubtract(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = val + d;
}
/* Kernel for adding two arrays */
__global__ void _kernel_ddmap2Add(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr)
{
	double val1, val2;
	getInputArrayValueForIndexingScheme(input1Arr, input1Offset, inputN, 0, &val1);
	getInputArrayValueForIndexingScheme(input2Arr, input2Offset, inputN, 0, &val2);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = val1 + val2;
}
/* Kernel for subtracting a constant from an array */
__global__ void _kernel_ddmapSubtract2(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = d - val;
}
/* Kernel for subtracting two arrays*/
__global__ void _kernel_ddmap2Subtract(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr)
{
	double val1, val2;
	getInputArrayValueForIndexingScheme(input1Arr, input1Offset, inputN, 0, &val1);
	getInputArrayValueForIndexingScheme(input2Arr, input2Offset, inputN, 0, &val2);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = val1 - val2;
}
/* Kernel for multiplying an array by a constant */
__global__ void _kernel_ddmapMultiply(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = val * d;
}
/* Kernel for multiplying two arrays */
__global__ void _kernel_ddmap2Multiply(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr)
{
	double val1, val2;
	getInputArrayValueForIndexingScheme(input1Arr, input1Offset, inputN, 0, &val1);
	getInputArrayValueForIndexingScheme(input2Arr, input2Offset, inputN, 0, &val2);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = val1 * val2;
}
/* Kernel for dividing an array by a constant */
__global__ void _kernel_ddmapDivide(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = val / d;
}
/* Kernel for dividing a constant by an array */
__global__ void _kernel_ddmapDivide2(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = d / val;
}
/* Kernel for dividing two arrays*/
__global__ void _kernel_ddmap2Divide(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr)
{
	double val1, val2;
	getInputArrayValueForIndexingScheme(input1Arr, input1Offset, inputN, 0, &val1);
	getInputArrayValueForIndexingScheme(input2Arr, input2Offset, inputN, 0, &val2);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = val1 / val2;
}
/* Kernel for raising an array to the power of a constant*/
__global__ void _kernel_ddmapPower(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = pow(val, d);
}
/* Kernel for raising a constant to the power of each array element */
__global__ void _kernel_ddmapPower2(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = pow(d, val);
}
/* Kernel for raising each element of one array to the power of one element in another */
__global__ void _kernel_ddmap2Power(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr)
{
	double val1, val2;
	getInputArrayValueForIndexingScheme(input1Arr, input1Offset, inputN, 0, &val1);
	getInputArrayValueForIndexingScheme(input2Arr, input2Offset, inputN, 0, &val2);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = pow(val1, val2);
}
/* Kernel for square rooting an array */
__global__ void _kernel_ddmapSqrt(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = sqrt(val);
}
/* Kernel for inverse cos of each element of an array */
__global__ void _kernel_ddmapArcCos(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = acos(val);
}
/* Kernel for cos of each element of an array */
__global__ void _kernel_ddmapCos(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = cos(val);
}
/* Kernel for hyperbolic cos of each element of an array */
__global__ void _kernel_ddmapCosh(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = cosh(val);
}

__global__ void _kernel_ddmapArcSin(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = asin(val);
}

__global__ void _kernel_ddmapSin(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = sin(val);
}

__global__ void _kernel_ddmapSinh(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = sinh(val);
}

__global__ void _kernel_ddmapArcTan(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = atan(val);
}

__global__ void _kernel_ddmapTan(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = tan(val);
}

__global__ void _kernel_ddmapTanh(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = tanh(val);
}

__global__ void _kernel_ddmapLog(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = log(val);
}

__global__ void _kernel_ddmapLog10(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = log10(val);
}

__global__ void _kernel_ddmapExp(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = exp(val);
}

/******************************************************************************************************************/
/* double to bool kernel maps */
/******************************************************************************************************************/

/* Kernel for calculating elementwise greater than value over constant and array */
__global__ void _kernel_dbmapGT(double *inputArr, const int inputOffset, const int inputN, const double d, int *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = val > d;
}

/* Kernel for calculating elementwise greater than value over array and constant */
__global__ void _kernel_dbmapGT2(double *inputArr, const int inputOffset, const int inputN, const double d, int *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = d > val;
}


/* Kernel for calculating elementwise greater than value over two arrays */
__global__ void _kernel_dbmap2GT(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, int *outputArr)
{
	double val1, val2;
	getInputArrayValueForIndexingScheme(input1Arr, input1Offset, inputN, 0, &val1);
	getInputArrayValueForIndexingScheme(input2Arr, input2Offset, inputN, 0, &val2);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = val1 > val2;
}

/* Kernel for calculating elementwise greater than or equal value over constant and array */
__global__ void _kernel_dbmapGTE(double *inputArr, const int inputOffset, const int inputN, const double d, int *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = val >= d;
}

/* Kernel for calculating elementwise greater than or equal value over array and constant */
__global__ void _kernel_dbmapGTE2(double *inputArr, const int inputOffset, const int inputN, const double d, int *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = d >= val;
}


/* Kernel for calculating elementwise greater than or equal value over two arrays */
__global__ void _kernel_dbmap2GTE(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, int *outputArr)
{
	double val1, val2;
	getInputArrayValueForIndexingScheme(input1Arr, input1Offset, inputN, 0, &val1);
	getInputArrayValueForIndexingScheme(input2Arr, input2Offset, inputN, 0, &val2);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = val1 >= val2;
}

/* Kernel for calculating elementwise less than value over array and constant */
__global__ void _kernel_dbmapLT(double *inputArr, const int inputOffset, const int inputN, const double d, int *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = val < d;
}

/* Kernel for calculating elementwise less than value over array and constant */
__global__ void _kernel_dbmapLT2(double *inputArr, const int inputOffset, const int inputN, const double d, int *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = d < val;
}

/* Kernel for calculating elementwise less than value over two arrays */
__global__ void _kernel_dbmap2LT(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, int *outputArr)
{
	double val1, val2;
	getInputArrayValueForIndexingScheme(input1Arr, input1Offset, inputN, 0, &val1);
	getInputArrayValueForIndexingScheme(input2Arr, input2Offset, inputN, 0, &val2);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = val1 < val2;
}

/* Kernel for calculating elementwise less than or equal value over constant and array */
__global__ void _kernel_dbmapLTE(double *inputArr, const int inputOffset, const int inputN, const double d, int *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = val <= d;
}

/* Kernel for calculating elementwise less than or equal value over array and constant */
__global__ void _kernel_dbmapLTE2(double *inputArr, const int inputOffset, const int inputN, const double d, int *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = d <= val;
}

/* Kernel for calculating elementwise less than or equal value over two arrays */
__global__ void _kernel_dbmap2LTE(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, int *outputArr)
{
	double val1, val2;
	getInputArrayValueForIndexingScheme(input1Arr, input1Offset, inputN, 0, &val1);
	getInputArrayValueForIndexingScheme(input2Arr, input2Offset, inputN, 0, &val2);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = val1 <= val2;
}

/* Kernel for calculating elementwise equality between array and constant */
__global__ void _kernel_dbmapEquality(double *inputArr, const int inputOffset, const int inputN, const double d, int *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = val == d;
}

/* Kernel for calculating elementwise equality over two arrays */
__global__ void _kernel_dbmap2Equality(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, int *outputArr)
{
	double val1, val2;
	getInputArrayValueForIndexingScheme(input1Arr, input1Offset, inputN, 0, &val1);
	getInputArrayValueForIndexingScheme(input2Arr, input2Offset, inputN, 0, &val2);
	outputArr[blockIdx.x * blockDim.x + threadIdx.x] = val1 == val2;
}

/******************************************************************************************************************/
/* double kernel reductions */
/******************************************************************************************************************/

/* Reduce to half the size */
__global__ void _kernel_ddreduceToHalf(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	double val;
	getInputArrayValueForIndexingScheme(inputArr, inputOffset, inputN, 0, &val);
	if ((blockIdx.x * blockDim.x + threadIdx.x) % 2 == 0) 
	{
		outputArr[(blockIdx.x * blockDim.x + threadIdx.x) / 2] = val;
	}
}

/******************************************************************************************************************/
/* double to double maps */
/******************************************************************************************************************/

int ddmapAdd(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapAddSubtract << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, d, outputArr);
	return cudaGetLastError();
}

int ddmap2Add(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmap2Add << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, inputN, outputArr);
	return cudaGetLastError();
}

int ddmapSubtract(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapAddSubtract << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, -d, outputArr);
	return cudaGetLastError();
}

int ddmapSubtract2(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapAddSubtract << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, d, outputArr);
	return cudaGetLastError();
}

int ddmap2Subtract(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmap2Subtract << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, inputN, outputArr);
	return cudaGetLastError();
}

int ddmapMultiply(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapMultiply << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, d, outputArr);
	return cudaGetLastError();
}

int ddmap2Multiply(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmap2Multiply << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, inputN, outputArr);
	return cudaGetLastError();
}

int ddmapDivide(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapDivide << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, d, outputArr);
	return cudaGetLastError();
}

int ddmapDivide2(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapDivide2 << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, d, outputArr);
	return cudaGetLastError();
}

int ddmap2Divide(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmap2Divide << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, inputN, outputArr);
	return cudaGetLastError();
}

int ddmapPower(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapPower << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, d, outputArr);
	return cudaGetLastError();
}

int ddmapPower2(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapPower2 << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, d, outputArr);
	return cudaGetLastError();
}

int ddmap2Power(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmap2Power << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, inputN, outputArr);
	return cudaGetLastError();
}

int ddmapSqrt(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapSqrt << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, outputArr);
	return cudaGetLastError();
}

int ddmapArcCos(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapArcCos << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, outputArr);
	return cudaGetLastError();
}

int ddmapCos(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapCos << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, outputArr);
	return cudaGetLastError();
}

int ddmapCosh(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapCosh << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, outputArr);
	return cudaGetLastError();
}

int ddmapArcSin(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapArcSin << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, outputArr);
	return cudaGetLastError();
}

int ddmapSin(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapSin << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, outputArr);
	return cudaGetLastError();
}

int ddmapSinh(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapSinh << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, outputArr);
	return cudaGetLastError();
}

/* Function for calculating elementwise atan over an array */
int mapArcTan(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapArcTan << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, outputArr);
	return cudaGetLastError();
}

/* Function for calculating elementwise tan over an array */
int ddmapTan(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapTan << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, outputArr);
	return cudaGetLastError();
}

/* Function for calculating elementwise tanh over an array */
int ddmapTanh(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapTanh << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, outputArr);
	return cudaGetLastError();
}

/* Function for calculating elementwise log over an array */
int ddmapLog(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapLog << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, outputArr);
	return cudaGetLastError();
}

/* Function for calculating elementwise log10 over an array */
int ddmapLog10(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddmapLog10 << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, outputArr);
	return cudaGetLastError();
}

/******************************************************************************************************************/
/* double to bool maps */
/******************************************************************************************************************/

/* Function for calculating elementwise greater than value over array and constant */
int dbmapGT(double *inputArr, const int inputOffset, const int inputN, const double d, int *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_dbmapGT << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, d, outputArr);
	return cudaGetLastError();
}

/* Function for calculating elementwise greater than value over array and constant */
int dbmapGT2(double *inputArr, const int inputOffset, const int inputN, const double d, int *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_dbmapGT2 << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, d, outputArr);
	return cudaGetLastError();
}

/* Function for calculating elementwise greater than value over two arrays */
int dbmap2GT(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, int *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_dbmap2GT << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, inputN, outputArr);
	return cudaGetLastError();
}

/* Function for calculating elementwise greater than or equal value over array and constant */
int dbmapGTE(double *inputArr, const int inputOffset, const int inputN, const double d, int *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_dbmapGTE << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, d, outputArr);
	return cudaGetLastError();
}

/* Function for calculating elementwise greater than or equal value over array and constant */
int dbmapGTE2(double *inputArr, const int inputOffset, const int inputN, const double d, int *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_dbmapGTE2 << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, d, outputArr);
	return cudaGetLastError();
}

/* Function for calculating elementwise greater than or equal over two arrays */
int dbmap2GTE(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, int *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_dbmap2GTE << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, inputN, outputArr);
	return cudaGetLastError();
}

/* Function for calculating elementwise less than value over array and constant */
int dbmapLT(double *inputArr, const int inputOffset, const int inputN, const double d, int *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_dbmapLT << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, d, outputArr);
	return cudaGetLastError();
}

/* Function for calculating elementwise less than value over array and constant */
int dbmapLT2(double *inputArr, const int inputOffset, const int inputN, const double d, int *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_dbmapLT2 << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, d, outputArr);
	return cudaGetLastError();
}

/* Function for calculating elementwise less then value over two arrays */
int dbmap2LT(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, int *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_dbmap2LT << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, inputN, outputArr);
	return cudaGetLastError();
}

/* Function for calculating elementwise less than or equal over array and constant */
int dbmapLTE(double *inputArr, const int inputOffset, const int inputN, const double d, int *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_dbmapLTE << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, d, outputArr);
	return cudaGetLastError();
}

/* Function for calculating elementwise less than or equal over array and constant */
int dbmapLTE2(double *inputArr, const int inputOffset, const int inputN, const double d, int *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_dbmapLTE2 << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, d, outputArr);
	return cudaGetLastError();
}

/* Function for calculating elementwise less then or equal over two arrays */
int dbmap2LTE(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, int *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_dbmap2LTE << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, inputN, outputArr);
	return cudaGetLastError();
}

/* Function for calculating elementwise equality over array and constant */
int dbmapEquality(double *inputArr, const int inputOffset, const int inputN, const double d, int *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_dbmapEquality << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, d, outputArr);
	return cudaGetLastError();
}

/* Function for calculating elementwise equality over two arrays */
int dbmap2Equality(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, int *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_dbmap2Equality << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, inputN, outputArr);
	return cudaGetLastError();
}

/******************************************************************************************************************/
/* double reductions */
/******************************************************************************************************************/

int ddreduceToHalf(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_ddreduceToHalf << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, inputN, outputArr);
	return cudaGetLastError();
}