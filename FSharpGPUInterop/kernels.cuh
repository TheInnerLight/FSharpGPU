/*This file is part of FSharpGPU.

	FSharpGPU is free software : you can redistribute it and / or modify
	it under the terms of the GNU Affero General Public License as
	published by the Free Software Foundation, either version 3 of the
	License, or(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
	GNU Affero General Public License for more details.

	You should have received a copy of the GNU Affero General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/* This software contains source code provided by NVIDIA Corporation. */

/*Copyright © 2015 Philip Curzon */

#pragma once

/* double to double maps */

extern "C" __declspec(dllexport) int ddmapAdd(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr);

extern "C" __declspec(dllexport) int ddmap2Add(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr);

extern "C" __declspec(dllexport) int ddmapSubtract(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr);

extern "C" __declspec(dllexport) int ddmapSubtract2(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr);

extern "C" __declspec(dllexport) int ddmap2Subtract(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr);

extern "C" __declspec(dllexport) int ddmapMultiply(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr);

extern "C" __declspec(dllexport) int ddmap2Multiply(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr);

extern "C" __declspec(dllexport) int ddmapDivide(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr);

extern "C" __declspec(dllexport) int ddmapDivide2(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr);

extern "C" __declspec(dllexport) int ddmap2Divide(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr);

extern "C" __declspec(dllexport) int ddmapPower(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr);

extern "C" __declspec(dllexport) int ddmapPower2(double *inputArr, const int inputOffset, const int inputN, const double d, double *outputArr);

extern "C" __declspec(dllexport) int ddmap2Power(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, double *outputArr);

extern "C" __declspec(dllexport) int ddmapSqrt(double *inputArr, const int inputOffset, const int inputN, double *outputArr);

extern "C" __declspec(dllexport) int ddmapArcCos(double *inputArr, const int inputOffset, const int inputN, double *outputArr);

extern "C" __declspec(dllexport) int ddmapCos(double *inputArr, const int inputOffset, const int inputN, double *outputArr);

extern "C" __declspec(dllexport) int ddmapCosh(double *inputArr, const int inputOffset, const int inputN, double *outputArr);

extern "C" __declspec(dllexport) int ddmapArcSin(double *inputArr, const int inputOffset, const int inputN, double *outputArr);

extern "C" __declspec(dllexport) int ddmapSin(double *inputArr, const int inputOffset, const int inputN, double *outputArr);

extern "C" __declspec(dllexport) int ddmapSinh(double *inputArr, const int inputOffset, const int inputN, double *outputArr);

extern "C" __declspec(dllexport) int ddmapArcTan(double *inputArr, const int inputOffset, const int inputN, double *outputArr);

extern "C" __declspec(dllexport) int ddmapTan(double *inputArr, const int inputOffset, const int inputN, double *outputArr);

extern "C" __declspec(dllexport) int ddmapTanh(double *inputArr, const int inputOffset, const int inputN, double *outputArr);

extern "C" __declspec(dllexport) int ddmapLog(double *inputArr, const int inputOffset, const int inputN, double *outputArr);

extern "C" __declspec(dllexport) int ddmapLog10(double *inputArr, const int inputOffset, const int inputN, double *outputArr);


/* double to bool maps */

extern "C" __declspec(dllexport) int dbmapGT(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr);

extern "C" __declspec(dllexport) int dbmapGT2(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr);

extern "C" __declspec(dllexport) int dbmap2GT(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, __int32 *outputArr);

extern "C" __declspec(dllexport) int dbmapGTE(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr);

extern "C" __declspec(dllexport) int dbmapGTE2(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr);

extern "C" __declspec(dllexport) int dbmap2GTE(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, __int32 *outputArr);

extern "C" __declspec(dllexport) int dbmapLT(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr);

extern "C" __declspec(dllexport) int dbmapLT2(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr);

extern "C" __declspec(dllexport) int dbmap2LT(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, __int32 *outputArr);

extern "C" __declspec(dllexport) int dbmapLTE(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr);

extern "C" __declspec(dllexport) int dbmapLTE2(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr);

extern "C" __declspec(dllexport) int dbmap2LTE(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, __int32 *outputArr);

extern "C" __declspec(dllexport) int dbmapEquality(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr);

extern "C" __declspec(dllexport) int dbmap2Equality(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, __int32 *outputArr);

extern "C" __declspec(dllexport) int dbmapNotEquality(double *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr);

extern "C" __declspec(dllexport) int dbmap2NotEquality(double *input1Arr, const int input1Offset, double *input2Arr, const int input2Offset, const int inputN, __int32 *outputArr);

/* bool to bool maps */

extern "C" __declspec(dllexport) int bbmapConditionAnd(__int32 *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr);

extern "C" __declspec(dllexport) int bbmap2ConditionAnd(__int32 *input1Arr, const int input1Offset, __int32 *input2Arr, const int input2Offset, const __int32 inputN, __int32 *outputArr);

extern "C" __declspec(dllexport) int bbmapConditionOr(__int32 *inputArr, const int inputOffset, const int inputN, const double d, __int32 *outputArr);

extern "C" __declspec(dllexport) int bbmap2ConditionOr(__int32 *input1Arr, const int input1Offset, __int32 *input2Arr, const int input2Offset, const __int32 inputN, __int32 *outputArr);

/* double reductions */

extern "C" __declspec(dllexport) int ddreduceToHalf(double *inputArr, const int inputOffset, const int inputN, double *outputArr);

extern "C" __declspec(dllexport) int ddsumTotal(double *inputArr, const int inputOffset, const int inputN, double *outputArr);

/* double filters */

extern "C" __declspec(dllexport) int ddfilter(double *inputArr, __int32 *predicateArr, const int inputN, double **outputArr, __int32 *outputN);

extern "C" __declspec(dllexport) int ddpartition(double *inputArr, __int32 *predicateArr, const int inputN, double **outputArrTrue, double **outputArrFalse, __int32 *outputNTrue, __int32 *outputNFalse);

/* mutation */

extern "C" __declspec(dllexport) int ddsetAll(double *arr, const int offset, const int n, const double value);