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

extern "C" __declspec(dllexport) int createCUDAArray(size_t n, size_t typeSize, void **devPtr);

extern "C" __declspec(dllexport) int freeCUDAArray(void *devPtr);

extern "C" __declspec(dllexport) int createCUDADoubleArray(const size_t arraySize, double **devPtr);

extern "C" __declspec(dllexport) int initialiseCUDADoubleArray(const double dblArray[], size_t n, double **devPtr);

extern "C" __declspec(dllexport) int retrieveCUDADoubleArray(double *devPtr, const size_t offset, double dblArray[], const size_t n);

extern "C" __declspec(dllexport) int createCUDABoolArray(size_t n, __int32 **devPtr);

extern "C" __declspec(dllexport) int initialiseCUDABoolArray(const __int32 *array, const size_t n, __int32 **devPtr);

extern "C" __declspec(dllexport) int retrieveCUDABoolArray(__int32 *devPtr, const size_t offset, __int32 dblArray[], const size_t n);