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

extern "C" __declspec(dllexport) int createCUDAArray(int n, int typeSize, void **devPtr);

extern "C" __declspec(dllexport) int freeCUDAArray(void *devPtr);

extern "C" __declspec(dllexport) int createCUDADoubleArray(const int arraySize, double **devPtr);

extern "C" __declspec(dllexport) int initialiseCUDADoubleArray(const double dblArray[], int n, double **devPtr);

extern "C" __declspec(dllexport) int retrieveCUDADoubleArray(double *devPtr, const int offset, double dblArray[], const int n);

extern "C" __declspec(dllexport) int createCUDABoolArray(int n, int **devPtr);

extern "C" __declspec(dllexport) int initialiseCUDABoolArray(const int *array, const int n, int **devPtr);

extern "C" __declspec(dllexport) int retrieveCUDABoolArray(int *devPtr, const int offset, int dblArray[], const int n);