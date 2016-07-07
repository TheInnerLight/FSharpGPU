(*This file is part of FSharpGPU.

FSharpGPU is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FSharpGPU is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FSharpGPU.  If not, see <http://www.gnu.org/licenses/>.
*)

(* Copyright © 2015 Philip Curzon *)

namespace NovelFS.FSharpGPU

open System
open System.Runtime.InteropServices

module internal GeneralDevice = 
    #if x64
    [<Literal>]
    let platformDLL = @"..\..\..\FSharpGPUInterop\Debug\x64\FSharpGPUInterop.dll"
#else
    [<Literal>]
    let platformDLL = @"..\..\..\FSharpGPUInterop\Debug\Win32\FSharpGPUInterop.dll"
#endif

    [<DllImport(platformDLL, EntryPoint="createCUDAArray", CallingConvention = CallingConvention.Cdecl)>]
    extern int createUninitialisedArray(int size, int typeSize, IntPtr& handle)

    [<DllImport(platformDLL, EntryPoint="freeCUDAArray", CallingConvention = CallingConvention.Cdecl)>]
    extern int freeArray(IntPtr handle)

    [<DllImport(platformDLL, EntryPoint="createCUDADoubleArray", CallingConvention = CallingConvention.Cdecl)>]
    extern int createUninitialisedCUDADoubleArray(int size, IntPtr& handle)

    [<DllImport(platformDLL, EntryPoint="initialiseCUDADoubleArray", CallingConvention = CallingConvention.Cdecl)>]
    extern int initialiseCUDADoubleArray(double[] flts, int n, IntPtr& handle)

    [<DllImport(platformDLL, EntryPoint="retrieveCUDADoubleArray", CallingConvention = CallingConvention.Cdecl)>]
    extern int retrieveCUDADoubleArray(IntPtr handle, int offset, double[] flts, int n)

    [<DllImport(platformDLL, EntryPoint="createCUDABoolArray", CallingConvention = CallingConvention.Cdecl)>]
    extern int createUninitialisedCUDABoolArray(int size, IntPtr& handle)

    [<DllImport(platformDLL, EntryPoint="initialiseCUDABoolArray", CallingConvention = CallingConvention.Cdecl)>]
    extern int initialiseCUDABoolArray(bool[] flts, int n, IntPtr& handle)

    [<DllImport(platformDLL, EntryPoint="retrieveCUDABoolArray", CallingConvention = CallingConvention.Cdecl)>]
    extern int retrieveCUDABoolArray(IntPtr handle, int offset, int[] flts, int n)

module internal DeviceFloatKernels = 

    // Float to Float mappings

    // Addition

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapAdd", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapAdd(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmap2Add", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2Add(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    // Subtraction

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapSubtract", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapSubtract(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapSubtract2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapSubtract2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmap2Subtract", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2Subtract(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    // Multiplication

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapMultiply", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapMultiply(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmap2Multiply", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2Multiply(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    // Division

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapDivide", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapDivide(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapDivide2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapDivide2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmap2Divide", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2Divide(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    // Power

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapPower", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapPower(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapPower2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapPower2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmap2Power", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2Power(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    // Various maths functions

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapSqrt", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapSqrt(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapArcCos", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapArcCos(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapCos", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapCos(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapCosh", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapCosh(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapArcSin", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapArcSin(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapSin", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapSin(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapSinh", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapSinh(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapArcTan", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapArcTan(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapTan", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapTan(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapTanh", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapTanh(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapLog", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapLog(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddmapLog10", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapLog10(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    // Float to Bool mappings

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="dbmapGT", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapGreaterThan(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="dbmapGT2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapGreaterThan2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="dbmap2GT", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2GreaterThan(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="dbmapGTE", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapGreaterThanOrEqual(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="dbmapGTE2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapGreaterThanOrEqual2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="dbmap2GTE", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2GreaterThanOrEqual(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="dbmapLT", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapLessThan(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="dbmapLT2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapLessThan2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="dbmap2LT", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2LessThan(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="dbmapLTE", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapLessThanOrEqual(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="dbmapLTE2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapLessThanOrEqual2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="dbmap2LTE", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2LessThanOrEqual(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="dbmapEquality", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapEquality(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="dbmap2Equality", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2Equality(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="dbmapNotEquality", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapInequality(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="dbmap2NotEquality", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2NotEquality(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    // Float to Float reductions

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddreduceToHalf", CallingConvention = CallingConvention.Cdecl)>]
    extern int reduceToHalf(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddsumTotal", CallingConvention = CallingConvention.Cdecl)>]
    extern int sumTotal(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    // Float to Float filters

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddfilter", CallingConvention = CallingConvention.Cdecl)>]
    extern int filter(IntPtr inArr, IntPtr predArr, int inputN, IntPtr& outArr, int& size)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddpartition", CallingConvention = CallingConvention.Cdecl)>]
    extern int partition(IntPtr inArr, IntPtr predArr, int inputN, IntPtr& outTrueArr, IntPtr& outFalseArr, int& trueSize, int& falseSize);

    // Float mutation

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="ddsetAll", CallingConvention = CallingConvention.Cdecl)>]
    extern int setAllElementsToConstant(IntPtr inArr, int inputOffset, int inputN, double value)

module internal DeviceBoolKernels = 

    // Bool to Bool maps

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="bbmapConditionAnd", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapConditionalAnd(IntPtr inArr, int inputOffset, int inputN, int bl, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="bbmap2ConditionAnd", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2ConditionalAnd(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="bbmapConditionOr", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapConditionalOr(IntPtr inArr, int inputOffset, int inputN, int bl, IntPtr outArr)

    [<DllImport(GeneralDevice.platformDLL, EntryPoint="bbmap2ConditionOr", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2ConditionalOr(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

