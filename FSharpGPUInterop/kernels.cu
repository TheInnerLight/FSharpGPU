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
#include "kernels.cuh"
#include "functions.cuh"
#include "scankernels.cuh"
#include "templated_kernels.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>

/* Split the size of the array between threads and blocks */
ThreadBlocks getThreadsAndBlocks(const int n)
{
	ThreadBlocks tb;
	tb.threadCount = std::min(MAX_THREADS, n);
	tb.blockCount = std::min(MAX_BLOCKS, std::max(1, (n + tb.threadCount - 1) / tb.threadCount));
	tb.thrBlockCount = tb.threadCount * tb.blockCount;
	tb.loopCount = std::max(1, (n + tb.thrBlockCount - 1) / tb.thrBlockCount);
	tb.N = n;
	return tb;
}

/* Split the size of the array between threads and blocks */
ThreadBlocks getThreadsAndBlocks32(const int n)
{
	ThreadBlocks tb;
	
	tb.threadCount = std::min(MAX_THREADS, n);
	tb.loopCount = std::min(32, std::max(1, (n + tb.threadCount - 1) / tb.threadCount));
	int thrLoopCount = tb.loopCount * tb.threadCount;
	tb.blockCount = std::min(MAX_BLOCKS, std::max(1, (n + thrLoopCount - 1) / thrLoopCount));
	tb.thrBlockCount = tb.threadCount * tb.blockCount;
	tb.N = n;
	return tb;
}

/******************************************************************************************************************/
/* double filters */
/******************************************************************************************************************/

/* Kernel for filtering double array based on boolean array predicate */
__global__ void _kernel_ddfilter(double *inputArr, __int32 *predicateArr, const ThreadBlocks inputN, __int32 *nres, double *outputArr)
{
	__shared__ __int32 l_n;

	for (int iter = 0; iter < inputN.loopCount; ++iter) {
		// zero the counter
		if (threadIdx.x == 0) l_n = 0;
		__syncthreads();

		// get the values of the array and the predicate
		double d;
		__int32 b, pos;

		//__int32 i = iter*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x;
		int i = (blockIdx.x * inputN.loopCount * inputN.threadCount) + iter * inputN.threadCount + threadIdx.x;

		if (i < inputN.N) {
			d = inputArr[i];
			b = predicateArr[i];
			if (b != 0)
				pos = atomicAdd(&l_n, 1); // increment the counter for those which are true
		}
		__syncthreads();

		// leader increments the global counter
		if (threadIdx.x == 0)
			l_n = atomicAdd(nres, l_n);
		__syncthreads();

		// threads with true predicates write their elements
		if (i < inputN.N && b != 0) {
			pos += l_n; // increment local pos by global counter
			outputArr[pos] = d;
		}
		__syncthreads();
	}
}

/* Kernel for filtering double array based a prefix counter */
__global__ void _kernel_ddfilterPrefix(double *inputArr, __int32 *prefixArr, const ThreadBlocks inputN, double *outputArr)
{
	for (int iter = 0; iter < inputN.loopCount; ++iter) {
		int i = iter*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x;
		if (i > 0 && i < inputN.N && prefixArr[i] > 0)
		{
			if (prefixArr[i - 1] < prefixArr[i]) 
			{
				outputArr[(prefixArr[i]-1)] = inputArr[i - 1];
			}
		}
	}
}

/* Kernel for filtering double array based a prefix counter */
__global__ void _kernel_ddinvFilterPrefix(double *inputArr, __int32 *prefixArr, const ThreadBlocks inputN, double *outputArr)
{
	for (int iter = 0; iter < inputN.loopCount; ++iter) {
		int i = iter*inputN.thrBlockCount + blockIdx.x * blockDim.x + threadIdx.x;
		if (i > 0 && i < inputN.N)
		{
			if (prefixArr[i - 1] >= prefixArr[i])
			{
				outputArr[i - 1 - prefixArr[i]] = inputArr[i - 1];
			}
		}
	}
}

/* Kernel for filtering double array based a prefix counter */
__global__ void _kernel_iiInit(__int32 *prefixArr, const ThreadBlocks inputN, __int32 val)
{
	for (int iter = 0; iter < inputN.loopCount; ++iter) {
		int i = (blockIdx.x * inputN.loopCount * inputN.threadCount) + iter * inputN.threadCount + threadIdx.x;
		if (i < inputN.N)
		{
			prefixArr[i] = val;
		}
	}
}

/******************************************************************************************************************/
/* double to double maps */
/******************************************************************************************************************/

// Declarations of generic device functions

typedef double(*dbl_func)(double);
typedef double(*dbl_dbl_func)(double, double);
typedef __int32(*dbl_int32_func)(double, double);
typedef __int32(*int32_int32_func)(__int32, __int32);
// arithmetic functions
__device__ dbl_dbl_func add_kernel = _kernel_add<double, double>;
__device__ dbl_dbl_func subtract_kernel = _kernel_subtract<double, double>;
__device__ dbl_dbl_func multiply_kernel = _kernel_multiply<double, double>;
__device__ dbl_dbl_func divide_kernel = _kernel_divide<double, double>;
__device__ dbl_dbl_func power_kernel = _kernel_power<double, double>;
// comparison functions
__device__ dbl_int32_func greater_than_kernel = _kernel_greater_than<double>;
__device__ dbl_int32_func greater_than_or_equal_kernel = _kernel_greater_than_or_equal<double>;
__device__ dbl_int32_func less_than_kernel = _kernel_less_than<double>;
__device__ dbl_int32_func less_than_or_equal_kernel = _kernel_less_than_or_equal<double>;
// equality funcitons
__device__ dbl_int32_func equality_kernel = _kernel_equality<double>;
__device__ dbl_int32_func inequality_kernel = _kernel_inequality<double>;
// conditional functions
__device__ int32_int32_func conditional_and_kernel = _kernel_conditional_and;
__device__ int32_int32_func conditional_or_kernel = _kernel_conditional_or;
// maths functions
__device__ dbl_func sqrt_kernel = _kernel_sqrt<double, double>;
__device__ dbl_func sin_kernel = _kernel_sin<double, double>;
__device__ dbl_func cos_kernel = _kernel_cos<double, double>;
__device__ dbl_func tan_kernel = _kernel_tan<double, double>;
__device__ dbl_func sinh_kernel = _kernel_sinh<double, double>;
__device__ dbl_func cosh_kernel = _kernel_cosh<double, double>;
__device__ dbl_func tanh_kernel = _kernel_tanh<double, double>;
__device__ dbl_func arcsin_kernel = _kernel_arcsin<double, double>;
__device__ dbl_func arccos_kernel = _kernel_arccos<double, double>;
__device__ dbl_func arctan_kernel = _kernel_arctan<double, double>;
__device__ dbl_func log_kernel = _kernel_log<double, double>;
__device__ dbl_func log10_kernel = _kernel_log10<double, double>;
__device__ dbl_func exp_kernel = _kernel_exp<double, double>;


// _kernel_map_op is for applying functions to an array element
// _kernel_map_with_const_op is for applying functions to an array element and a fixed value
// _kernel_map_with_const_op2 is for non-commutative functions and has the opposite ordering to _kernel_map_with_const_op
// _kernel_map2_op is for applying function which takes two array elements as arguments

/* Function for adding an array to a constant */
int ddmapAdd(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_dbl_func add_kernel_h;
	cudaMemcpyFromSymbol(&add_kernel_h, add_kernel, sizeof(dbl_dbl_func));
	_kernel_map_with_const_op<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, add_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for adding two arrays */
int ddmap2Add(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_dbl_func add_kernel_h;
	cudaMemcpyFromSymbol(&add_kernel_h, add_kernel, sizeof(dbl_dbl_func));
	_kernel_map2_op<double, double> << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, tb, outputArr, add_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for subtracting an array and a constant */
int ddmapSubtract(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_dbl_func subtract_kernel_h;
	cudaMemcpyFromSymbol(&subtract_kernel_h, subtract_kernel, sizeof(dbl_dbl_func));
	_kernel_map_with_const_op<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, subtract_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for subtracting a constant and an array */
int ddmapSubtract2(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_dbl_func subtract_kernel_h;
	cudaMemcpyFromSymbol(&subtract_kernel_h, subtract_kernel, sizeof(dbl_dbl_func));
	_kernel_map_with_const_op2<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, subtract_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for subtracting two arrays */
int ddmap2Subtract(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_dbl_func subtract_kernel_h;
	cudaMemcpyFromSymbol(&subtract_kernel_h, subtract_kernel, sizeof(dbl_dbl_func));
	_kernel_map2_op<double, double> << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, tb, outputArr, subtract_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for multiplying an array and a constant */
int ddmapMultiply(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_dbl_func multiply_kernel_h;
	cudaMemcpyFromSymbol(&multiply_kernel_h, multiply_kernel, sizeof(dbl_dbl_func));
	_kernel_map_with_const_op<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, multiply_kernel_h, ONE);
	return cudaGetLastError();
}

/* Function for multiplying two arrays */
int ddmap2Multiply(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_dbl_func multiply_kernel_h;
	cudaMemcpyFromSymbol(&multiply_kernel_h, multiply_kernel, sizeof(dbl_dbl_func));
	_kernel_map2_op<double, double> << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, tb, outputArr, multiply_kernel_h, ONE);
	return cudaGetLastError();
}

/* Function for dividing an array and a constant */
int ddmapDivide(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_dbl_func divide_kernel_h;
	cudaMemcpyFromSymbol(&divide_kernel_h, divide_kernel, sizeof(dbl_dbl_func));
	_kernel_map_with_const_op<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, divide_kernel_h, ONE);
	return cudaGetLastError();
}

/* Function for dividing a constant and an array */
int ddmapDivide2(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_dbl_func divide_kernel_h;
	cudaMemcpyFromSymbol(&divide_kernel_h, divide_kernel, sizeof(dbl_dbl_func));
	_kernel_map_with_const_op2<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, divide_kernel_h, ONE);
	return cudaGetLastError();
}

/* Function for dividing two arrays */
int ddmap2Divide(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_dbl_func divide_kernel_h;
	cudaMemcpyFromSymbol(&divide_kernel_h, divide_kernel, sizeof(dbl_dbl_func));
	_kernel_map2_op<double, double> << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, tb, outputArr, divide_kernel_h, ONE);
	return cudaGetLastError();
}

/* Function for raising an array element to the power of a constant */
__int32 ddmapPower(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_dbl_func power_kernel_h;
	cudaMemcpyFromSymbol(&power_kernel_h, power_kernel, sizeof(dbl_dbl_func));
	_kernel_map_with_const_op<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, power_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for raising a constant to the power of an array element */
int ddmapPower2(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_dbl_func power_kernel_h;
	cudaMemcpyFromSymbol(&power_kernel_h, power_kernel, sizeof(dbl_dbl_func));
	_kernel_map_with_const_op2<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, power_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for raising one array to the power of another */
int ddmap2Power(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_dbl_func power_kernel_h;
	cudaMemcpyFromSymbol(&power_kernel_h, power_kernel, sizeof(dbl_dbl_func));
	_kernel_map2_op<double, double> << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, tb, outputArr, power_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for elementwise square root over an array */
int ddmapSqrt(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_func sqrt_kernel_h;
	cudaMemcpyFromSymbol(&sqrt_kernel_h, sqrt_kernel, sizeof(dbl_func));
	_kernel_map_op<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, outputArr, sqrt_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for elementwise inverse cos over an array */
__int32 ddmapArcCos(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_func arccos_kernel_h;
	cudaMemcpyFromSymbol(&arccos_kernel_h, arccos_kernel, sizeof(dbl_func));
	_kernel_map_op<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, outputArr, arccos_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for elementwise cos over an array */
int ddmapCos(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_func cos_kernel_h;
	cudaMemcpyFromSymbol(&cos_kernel_h, cos_kernel, sizeof(dbl_func));
	_kernel_map_op<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, outputArr, cos_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for elementwise cosh over an array */
int ddmapCosh(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_func cosh_kernel_h;
	cudaMemcpyFromSymbol(&cosh_kernel_h, cosh_kernel, sizeof(dbl_func));
	_kernel_map_op<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, outputArr, cosh_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for elementwise inverse sin over an array */
int ddmapArcSin(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_func arcsin_kernel_h;
	cudaMemcpyFromSymbol(&arcsin_kernel_h, arcsin_kernel, sizeof(dbl_func));
	_kernel_map_op<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, outputArr, arcsin_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for elementwise sin of over an array */
int ddmapSin(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_func sin_kernel_h;
	cudaMemcpyFromSymbol(&sin_kernel_h, sin_kernel, sizeof(dbl_func));
	_kernel_map_op<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, outputArr, sin_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for elementwise sinh over an array */
int ddmapSinh(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_func sinh_kernel_h;
	cudaMemcpyFromSymbol(&sinh_kernel_h, sin_kernel, sizeof(dbl_func));
	_kernel_map_op<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, outputArr, sinh_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise atan over an array */
int mapArcTan(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_func arctan_kernel_h;
	cudaMemcpyFromSymbol(&arctan_kernel_h, arctan_kernel, sizeof(dbl_func));
	_kernel_map_op<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, outputArr, arctan_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise tan over an array */
int ddmapTan(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_func tan_kernel_h;
	cudaMemcpyFromSymbol(&tan_kernel_h, tan_kernel, sizeof(dbl_func));
	_kernel_map_op<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, outputArr, tan_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise tanh over an array */
int ddmapTanh(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_func tanh_kernel_h;
	cudaMemcpyFromSymbol(&tanh_kernel_h, tanh_kernel, sizeof(dbl_func));
	_kernel_map_op<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, outputArr, tanh_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise log over an array */
int ddmapLog(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_func log_kernel_h;
	cudaMemcpyFromSymbol(&log_kernel_h, log_kernel, sizeof(dbl_func));
	_kernel_map_op<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, outputArr, log_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise log10 over an array */
int ddmapLog10(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_func log10_kernel_h;
	cudaMemcpyFromSymbol(&log10_kernel_h, log10_kernel, sizeof(dbl_func));
	_kernel_map_op<double, double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, outputArr, log10_kernel_h, ZERO);
	return cudaGetLastError();
}

/******************************************************************************************************************/
/* double to bool maps */
/******************************************************************************************************************/

/* Function for calculating elementwise greater than value over array and constant */
int dbmapGT(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_int32_func greater_than_kernel_h;
	cudaMemcpyFromSymbol(&greater_than_kernel_h, greater_than_kernel, sizeof(dbl_int32_func));
	_kernel_map_with_const_op<double, __int32> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, greater_than_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise greater than value over array and constant */
int dbmapGT2(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_int32_func greater_than_kernel_h;
	cudaMemcpyFromSymbol(&greater_than_kernel_h, greater_than_kernel, sizeof(dbl_int32_func));
	_kernel_map_with_const_op2<double, __int32> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, greater_than_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise greater than value over two arrays */
int dbmap2GT(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_int32_func greater_than_kernel_h;
	cudaMemcpyFromSymbol(&greater_than_kernel_h, greater_than_kernel, sizeof(dbl_int32_func));
	_kernel_map2_op<double, __int32> << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, tb, outputArr, greater_than_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise greater than or equal value over array and constant */
int dbmapGTE(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_int32_func greater_than_or_equal_kernel_h;
	cudaMemcpyFromSymbol(&greater_than_or_equal_kernel_h, greater_than_or_equal_kernel, sizeof(dbl_int32_func));
	_kernel_map_with_const_op<double, __int32> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, greater_than_or_equal_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise greater than or equal value over array and constant */
int dbmapGTE2(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_int32_func greater_than_or_equal_kernel_h;
	cudaMemcpyFromSymbol(&greater_than_or_equal_kernel_h, greater_than_or_equal_kernel, sizeof(dbl_int32_func));
	_kernel_map_with_const_op2<double, __int32> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, greater_than_or_equal_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise greater than or equal over two arrays */
int dbmap2GTE(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_int32_func greater_than_or_equal_kernel_h;
	cudaMemcpyFromSymbol(&greater_than_or_equal_kernel_h, greater_than_or_equal_kernel, sizeof(dbl_int32_func));
	_kernel_map2_op<double, __int32> << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, tb, outputArr, greater_than_or_equal_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise less than value over array and constant */
int dbmapLT(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_int32_func less_than_kernel_h;
	cudaMemcpyFromSymbol(&less_than_kernel_h, less_than_kernel, sizeof(dbl_int32_func));
	_kernel_map_with_const_op<double, __int32> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, less_than_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise less than value over array and constant */
int dbmapLT2(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_int32_func less_than_kernel_h;
	cudaMemcpyFromSymbol(&less_than_kernel_h, less_than_kernel, sizeof(dbl_int32_func));
	_kernel_map_with_const_op2<double, __int32> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, less_than_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise less then value over two arrays */
int dbmap2LT(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_int32_func less_than_kernel_h;
	cudaMemcpyFromSymbol(&less_than_kernel_h, less_than_kernel, sizeof(dbl_int32_func));
	_kernel_map2_op<double, __int32> << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, tb, outputArr, less_than_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise less than or equal over array and constant */
__int32 dbmapLTE(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_int32_func less_than_kernel_or_equal_h;
	cudaMemcpyFromSymbol(&less_than_kernel_or_equal_h, less_than_or_equal_kernel, sizeof(dbl_int32_func));
	_kernel_map_with_const_op<double, __int32> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, less_than_kernel_or_equal_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise less than or equal over array and constant */
int dbmapLTE2(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_int32_func less_than_kernel_or_equal_h;
	cudaMemcpyFromSymbol(&less_than_kernel_or_equal_h, less_than_or_equal_kernel, sizeof(dbl_int32_func));
	_kernel_map_with_const_op2<double, __int32> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, less_than_kernel_or_equal_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise less then or equal over two arrays */
int dbmap2LTE(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_int32_func less_than_kernel_or_equal_h;
	cudaMemcpyFromSymbol(&less_than_kernel_or_equal_h, less_than_or_equal_kernel, sizeof(dbl_int32_func));
	_kernel_map2_op<double, __int32> << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, tb, outputArr, less_than_kernel_or_equal_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise equality over array and constant */
int dbmapEquality(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_int32_func equality_kernel_h;
	cudaMemcpyFromSymbol(&equality_kernel_h, equality_kernel, sizeof(dbl_int32_func));
	_kernel_map_with_const_op<double, __int32> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, equality_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise equality over two arrays */
int dbmap2Equality(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_int32_func equality_kernel_h;
	cudaMemcpyFromSymbol(&equality_kernel_h, equality_kernel, sizeof(dbl_int32_func));
	_kernel_map2_op<double, __int32> << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, tb, outputArr, equality_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise not equality over array and constant */
int dbmapNotEquality(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_int32_func inequality_kernel_h;
	cudaMemcpyFromSymbol(&inequality_kernel_h, inequality_kernel, sizeof(dbl_int32_func));
	_kernel_map_with_const_op<double, __int32> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, inequality_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise not equality over two arrays */
int dbmap2NotEquality(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const __int32 inputN, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	dbl_int32_func inequality_kernel_h;
	cudaMemcpyFromSymbol(&inequality_kernel_h, inequality_kernel, sizeof(dbl_int32_func));
	_kernel_map2_op<double, __int32> << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, tb, outputArr, inequality_kernel_h, ZERO);
	return cudaGetLastError();
}

/******************************************************************************************************************/
/* bool to bool kernel maps */
/******************************************************************************************************************/

/* Function for calculating elementwise conditional AND over array and constant */
int bbmapConditionAnd(__int32 *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	int32_int32_func conditional_and_kernel_h;
	cudaMemcpyFromSymbol(&conditional_and_kernel_h, conditional_and_kernel, sizeof(int32_int32_func));
	_kernel_map_with_const_op<__int32, __int32> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, conditional_and_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise conditional AND over two arrays */
int bbmap2ConditionAnd(__int32 *input1Arr, const int input1Offset, __int32 *input2Arr, const int input2Offset, const __int32 inputN, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	int32_int32_func conditional_and_kernel_h;
	cudaMemcpyFromSymbol(&conditional_and_kernel_h, conditional_and_kernel, sizeof(int32_int32_func));
	_kernel_map2_op<__int32, __int32> << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, tb, outputArr, conditional_and_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise conditional OR over array and constant */
int bbmapConditionOr(__int32 *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	int32_int32_func conditional_or_kernel_h;
	cudaMemcpyFromSymbol(&conditional_or_kernel_h, conditional_or_kernel, sizeof(int32_int32_func));
	_kernel_map_with_const_op<__int32, __int32> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, d, outputArr, conditional_or_kernel_h, ZERO);
	return cudaGetLastError();
}

/* Function for calculating elementwise conditional OR over two arrays */
int bbmap2ConditionOr(__int32 *input1Arr, const int input1Offset, __int32 *input2Arr, const int input2Offset, const __int32 inputN, __int32 *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	int32_int32_func conditional_or_kernel_h;
	cudaMemcpyFromSymbol(&conditional_or_kernel_h, conditional_or_kernel, sizeof(int32_int32_func));
	_kernel_map2_op<__int32, __int32> << < tb.blockCount, tb.threadCount >> >(input1Arr, input1Offset, input2Arr, input2Offset, tb, outputArr, conditional_or_kernel_h, ZERO);
	return cudaGetLastError();
}



/******************************************************************************************************************/
/* double reductions */
/******************************************************************************************************************/

int ddreduceToHalf(double *inputArr, const int inputOffset, const int inputN, double *outputArr)
{
	ThreadBlocks tb = getThreadsAndBlocks(inputN);
	_kernel_reduce_to_half<double> << < tb.blockCount, tb.threadCount >> >(inputArr, inputOffset, tb, outputArr);
	return cudaGetLastError();
}

/* Function for summing all elements in an array */
int ddsumTotal(double *inputArr, const int inputOffset, const int inputN, double *outputArr) 
{
	size_t nextPow2N = pow(2, ceil(log2(inputN)));
	double *workingArray, *workingArray2;
	createCUDADoubleArray(nextPow2N, &workingArray);
	createCUDADoubleArray(nextPow2N, &workingArray2);
	cudaMemset(workingArray, 0, nextPow2N*sizeof(double));
	cudaMemcpy(workingArray, inputArr + inputOffset, inputN*sizeof(double), cudaMemcpyDeviceToDevice);
	for (size_t curSize = nextPow2N; curSize > 1; curSize /= 2) 
	{
		ThreadBlocks tb = getThreadsAndBlocks(curSize);
		_kernel_sum_total<double> << < tb.blockCount, tb.threadCount >> >(workingArray, tb, workingArray2);
		double* temp = workingArray2;
		workingArray2 = workingArray;
		workingArray = temp;
	}

	cudaMemcpy(outputArr, workingArray, sizeof(double), cudaMemcpyDeviceToDevice);
	freeCUDAArray(workingArray);
	freeCUDAArray(workingArray2);
	return cudaGetLastError();
}

/******************************************************************************************************************/
/* double filters */
/******************************************************************************************************************/

/* Function for filtering a double array by a boolean array predicate */
int ddfilter(double *inputArr, __int32 *predicateArr, const int inputN, double **outputArr, __int32 *outputN)
{
	__int32 *prefixSum;
	int nP1 = inputN + 1; // prefix sum is one longer than array because of leading -1
	ThreadBlocks tb = getThreadsAndBlocks(nP1);
	
	cudaMalloc(&prefixSum, nP1 * sizeof(__int32));
	// Calculate parallel prefix sum
	ScanBlockAllocation sba = preallocBlockSums(nP1);
	prescanArray(prefixSum, predicateArr, nP1, sba);
	deallocBlockSums(sba);
	// copy length of array
	cudaMemcpy(outputN, prefixSum + inputN, sizeof(__int32), cudaMemcpyDeviceToHost);
	// alloc array of correct size
	createCUDADoubleArray(*outputN, outputArr);
	// filter using prefix sum
	_kernel_ddfilterPrefix << < tb.blockCount, tb.threadCount >> >(inputArr, prefixSum, tb, *outputArr);
	// cleanup
	cudaFree(prefixSum);
	return cudaGetLastError();
}

/* Function for partitioning a double array by a boolean array predicate */
int ddpartition(double *inputArr, __int32 *predicateArr, const int inputN, double **outputArrTrue, double **outputArrFalse, __int32 *outputNTrue, __int32 *outputNFalse)
{
	__int32 *prefixSum;
	int nP1 = inputN + 1; // prefix sum is one longer than array because of leading -1
	ThreadBlocks tb = getThreadsAndBlocks(nP1);

	cudaMalloc(&prefixSum, nP1 * sizeof(int));
	// Calculate parallel prefix sum
	ScanBlockAllocation sba = preallocBlockSums(nP1);
	prescanArray(prefixSum, predicateArr, nP1, sba);
	deallocBlockSums(sba);
	
	// copy length of arrays
	cudaMemcpy(outputNTrue, prefixSum + (inputN), sizeof(int), cudaMemcpyDeviceToHost);
	*outputNFalse = inputN - *outputNTrue;
	// alloc array of correct size
	createCUDADoubleArray(*outputNTrue, outputArrTrue);
	createCUDADoubleArray(*outputNFalse, outputArrFalse);
	// filter using prefix sum
	_kernel_ddfilterPrefix << < tb.blockCount, tb.threadCount >> >(inputArr, prefixSum, tb, *outputArrTrue);
	_kernel_ddinvFilterPrefix << < tb.blockCount, tb.threadCount >> >(inputArr, prefixSum, tb, *outputArrFalse);

	// cleanup
	cudaFree(prefixSum);
	return cudaGetLastError();
}

/******************************************************************************************************************/
/* mutation */
/******************************************************************************************************************/

/* Function for setting all the elements to a constant value */
int ddsetAll(double *arr, const int offset, const int n, const double value)
{
	ThreadBlocks tb = getThreadsAndBlocks(n);
	_kernel_set_all_elements_to_constant<double> << < tb.blockCount, tb.threadCount >> >(arr, offset, tb, value);
	return cudaGetLastError();
}