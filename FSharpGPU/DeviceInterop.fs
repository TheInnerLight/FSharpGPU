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

module internal DeviceFloatKernels = 

    // Float to Float mappings

    // Addition

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapAdd", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapAdd(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmap2Add", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2Add(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    // Subtraction

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapSubtract", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapSubtract(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapSubtract2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapSubtract2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmap2Subtract", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2Subtract(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    // Multiplication

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapMultiply", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapMultiply(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmap2Multiply", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2Multiply(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    // Division

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapDivide", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapDivide(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapDivide2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapDivide2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmap2Divide", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2Divide(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    // Power

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapPower", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapPower(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapPower2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapPower2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmap2Power", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2Power(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    // Various maths functions

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapSqrt", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapSqrt(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapArcCos", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapArcCos(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapCos", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapCos(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapCosh", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapCosh(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapArcSin", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapArcSin(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapSin", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapSin(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapSinh", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapSinh(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapArcTan", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapArcTan(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapTan", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapTan(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapTanh", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapTanh(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapLog", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapLog(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddmapLog10", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapLog10(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    // Float to Bool mappings

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="dbmapGT", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapGreaterThan(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="dbmapGT2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapGreaterThan2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="dbmap2GT", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2GreaterThan(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="dbmapGTE", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapGreaterThanOrEqual(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="dbmapGTE2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapGreaterThanOrEqual2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="dbmap2GTE", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2GreaterThanOrEqual(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="dbmapLT", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapLessThan(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="dbmapLT2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapLessThan2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="dbmap2LT", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2LessThan(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="dbmapLTE", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapLessThanOrEqual(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="dbmapLTE2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapLessThanOrEqual2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="dbmap2LTE", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2LessThanOrEqual(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="dbmapEquality", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapEquality(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="dbmap2Equality", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2Equality(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="dbmapNotEquality", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapInequality(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="dbmap2NotEquality", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2NotEquality(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    // Float to Float reductions

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddreduceToHalf", CallingConvention = CallingConvention.Cdecl)>]
    extern int reduceToHalf(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddsumTotal", CallingConvention = CallingConvention.Cdecl)>]
    extern int sumTotal(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    // Float to Float filters

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="ddfilter", CallingConvention = CallingConvention.Cdecl)>]
    extern int filter(IntPtr inArr, IntPtr predArr, int inputN, IntPtr outArr, int& size)

module internal DeviceBoolKernels = 

    // Bool to Bool maps

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="bbmapConditionAnd", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapConditionalAnd(IntPtr inArr, int inputOffset, int inputN, int bl, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="bbmap2ConditionAnd", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2ConditionalAnd(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="bbmapConditionOr", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapConditionalOr(IntPtr inArr, int inputOffset, int inputN, int bl, IntPtr outArr)

    [<DllImport(DeviceInterop.platformDLL, EntryPoint="bbmap2ConditionOr", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2ConditionalOr(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

module internal GeneralDeviceKernels = 
    /// A (type preserving) map function that involves a device array and a constant
    let private typePreservingMapWithConst cudaMapOperation constant (cudaArray : ComputeArray) =
        let mutable cudaPtr = System.IntPtr(0)
        DeviceInterop.createUninitialisedArray(cudaArray.Length, DeviceArrayInfo.length cudaArray.ArrayType, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
        let resultArray = ComputeArray(cudaArray.ArrayType, cudaPtr, cudaArray.Length, FullArray, AutoGenerated)
        cudaMapOperation (cudaArray.CudaPtr, cudaArray.Offset, cudaArray.Length, constant, resultArray.CudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
        resultArray
    /// A map function that involves a device array and a constant
    let private typeChangingMapWithConst cudaMapOperation constant (cudaArray : ComputeArray) newType =
        let mutable cudaPtr = System.IntPtr(0)
        DeviceInterop.createUninitialisedArray(cudaArray.Length, DeviceArrayInfo.length newType, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
        let resultArray = ComputeArray(newType, cudaPtr, cudaArray.Length, FullArray, AutoGenerated)
        cudaMapOperation (cudaArray.CudaPtr, cudaArray.Offset, cudaArray.Length, constant, resultArray.CudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
        resultArray
    /// A map function that involves only a device array
    let private typeChangingMap cudaMapOperation (cudaArray : ComputeArray) newType =
        let mutable cudaPtr = System.IntPtr(0)
        DeviceInterop.createUninitialisedArray(cudaArray.Length, DeviceArrayInfo.length newType, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
        let resultArray = ComputeArray(newType, cudaPtr, cudaArray.Length, FullArray, AutoGenerated)
        cudaMapOperation (cudaArray.CudaPtr, cudaArray.Offset, cudaArray.Length, resultArray.CudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
        resultArray
    /// A (type preserving) map function that involves only a device array
    let private typePreservingMap cudaMapOperation (cudaArray : ComputeArray) =
        let mutable cudaPtr = System.IntPtr(0)
        DeviceInterop.createUninitialisedArray(cudaArray.Length, DeviceArrayInfo.length cudaArray.ArrayType, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
        let resultArray = ComputeArray(cudaArray.ArrayType, cudaPtr, cudaArray.Length, FullArray, AutoGenerated)
        cudaMapOperation (cudaArray.CudaPtr, cudaArray.Offset, cudaArray.Length, resultArray.CudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
        resultArray
    /// A map2 function that involves two device arrays
    let private typeChangingMap2 cudaMap2Operation (cudaArray1 : ComputeArray) (cudaArray2 : ComputeArray) newType =
        let mutable cudaPtr = System.IntPtr(0)
        DeviceInterop.createUninitialisedArray(cudaArray1.Length, DeviceArrayInfo.length newType, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
        let resultArray = ComputeArray(newType, cudaPtr, cudaArray1.Length, FullArray, AutoGenerated)
        cudaMap2Operation (cudaArray1.CudaPtr, cudaArray1.Offset, cudaArray2.CudaPtr, cudaArray2.Offset, cudaArray1.Length, resultArray.CudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
        resultArray
    /// A (type preserving) map2 function that involves two device arrays
    let private typePreservingMap2 cudaMap2Operation (cudaArray1 : ComputeArray) (cudaArray2 : ComputeArray) =
        let mutable cudaPtr = System.IntPtr(0)
        DeviceInterop.createUninitialisedArray(cudaArray1.Length, DeviceArrayInfo.length cudaArray1.ArrayType, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
        let resultArray = ComputeArray(cudaArray1.ArrayType, cudaPtr, cudaArray1.Length, FullArray, AutoGenerated)
        cudaMap2Operation (cudaArray1.CudaPtr, cudaArray1.Offset, cudaArray2.CudaPtr, cudaArray2.Offset, cudaArray1.Length, resultArray.CudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
        resultArray
    /// A helper function for folding over cases useful for creating non-commutative arithmetic functions which can operate on many device types
    let private foldNonCommutativeArithmetic cmpVal1 cmpVal2 
        opFltVV opFltAV opFltVA opFltAA = // float value & float value, array & float value, float value & array, array & array
            match (cmpVal1, cmpVal2) with 
            |ResComputeFloat d1, ResComputeFloat d2 -> ResComputeFloat(opFltVV d1 d2)
            |ResComputeFloat d, ResComputeArray arr -> ResComputeArray(typePreservingMapWithConst (opFltVA) d arr)
            |ResComputeArray arr, ResComputeFloat d -> ResComputeArray(typePreservingMapWithConst (opFltAV) d arr)
            |ResComputeArray arr1, ResComputeArray arr2 -> ResComputeArray(typePreservingMap2 (opFltAA) arr1 arr2)
            |_ -> raise <| System.NotSupportedException()
    /// A helper function for folding over cases useful for creating commutative arithmetic functions which can operate on many device types
    let private foldCommutativeArithmetic cmpVal1 cmpVal2 opFltVV opFltAV opFltAA =
        foldNonCommutativeArithmetic cmpVal1 cmpVal2 
            opFltVV opFltAV opFltAV opFltAA // Float Operations
    /// A helper function for folding over cases useful for creating non-commutative conditional functions which can operate on many device types
    let private foldNonCommutativeConditional cmpVal1 cmpVal2 
        opBlVV opBlAV opBlVA opBlAA = // float value & float value, array & float value, float value & array, array & array
            match (cmpVal1, cmpVal2) with 
            |ResComputeBool d1, ResComputeBool d2 -> ResComputeBool(opBlVV d1 d2)
            |ResComputeBool b, ResComputeArray arr -> ResComputeArray(typePreservingMapWithConst (opBlAV) (System.Convert.ToInt32 b) arr)
            |ResComputeArray arr, ResComputeBool b -> ResComputeArray(typePreservingMapWithConst (opBlVA) (System.Convert.ToInt32 b) arr)
            |ResComputeArray arr1, ResComputeArray arr2 -> ResComputeArray(typePreservingMap2 (opBlAA) arr1 arr2)
            |_ -> raise <| System.NotSupportedException()
    /// A helper function for folding over cases useful for creating commutative conditional functions which can operate on many device types
    let private foldCommutativeConditional cmpVal1 cmpVal2 opFltVV opFltAV opFltAA =
        foldNonCommutativeConditional cmpVal1 cmpVal2 
            opFltVV opFltAV opFltAV opFltAA // Float Operations
    // A helper function for folding over cases useful for creating 1 argument maths functions with can operate on many device types
    let private fold1ArgMaths cmpVal
        opFltV opFltA =
            match cmpVal with
            |ResComputeFloat d -> ResComputeFloat(opFltV d)
            |ResComputeArray arr -> ResComputeArray(typePreservingMap opFltA arr)
            |_ -> raise <| System.NotSupportedException()

    // Arithmetic
    // ----------

    /// A function for elementwise addition of device elements
    let mapAdd cmpVal1 cmpVal2 =
        foldCommutativeArithmetic cmpVal1 cmpVal2 
            ( + ) (DeviceFloatKernels.mapAdd) (DeviceFloatKernels.map2Add) // Float Operations
    /// A function for elementwise subtraction of device elements
    let mapSubtract cmpVal1 cmpVal2 =
        foldNonCommutativeArithmetic cmpVal1 cmpVal2 
            ( - ) (DeviceFloatKernels.mapSubtract) (DeviceFloatKernels.mapSubtract2) (DeviceFloatKernels.map2Subtract) // Float Operations
    /// A function for elementwise multiplication of device elements
    let mapMultiply cmpVal1 cmpVal2 =
        foldCommutativeArithmetic cmpVal1 cmpVal2 
            ( * ) (DeviceFloatKernels.mapMultiply) (DeviceFloatKernels.map2Multiply) // Float Operations
    /// A function for elementwise division of device elements
    let mapDivide cmpVal1 cmpVal2 =
        foldNonCommutativeArithmetic cmpVal1 cmpVal2 
            ( / ) (DeviceFloatKernels.mapDivide) (DeviceFloatKernels.mapDivide2) (DeviceFloatKernels.map2Divide) // Float Operations
    /// A function for elementwise power raising of device elements
    let mapPower cmpVal1 cmpVal2 =
        foldNonCommutativeArithmetic cmpVal1 cmpVal2 
            ( ** ) (DeviceFloatKernels.mapPower) (DeviceFloatKernels.mapPower2) (DeviceFloatKernels.map2Power) // Float Operations

    // Maths functions
    // ---------------

    /// A function for elementwise sqrt of device elements
    let mapSqrt cmpVal =
        fold1ArgMaths cmpVal (sqrt) (DeviceFloatKernels.mapSqrt)
    /// A function for elementwise sin of device elements
    let mapSin cmpVal =
        fold1ArgMaths cmpVal (sin) (DeviceFloatKernels.mapSin)
    /// A function for elementwise cos of device elements
    let mapCos cmpVal =
        fold1ArgMaths cmpVal (cos) (DeviceFloatKernels.mapCos)
    /// A function for elementwise tan of device elements
    let mapTan cmpVal =
        fold1ArgMaths cmpVal (tan) (DeviceFloatKernels.mapTan)
    /// A function for elementwise hyperbolic sin of device elements
    let mapSinh cmpVal =
        fold1ArgMaths cmpVal (sinh) (DeviceFloatKernels.mapSinh)
    /// A function for elementwise hyperbolic cos of device elements
    let mapCosh cmpVal =
        fold1ArgMaths cmpVal (cosh) (DeviceFloatKernels.mapCosh)
    /// A function for elementwise hyperbolic tan of device elements
    let mapTanh cmpVal =
        fold1ArgMaths cmpVal (tanh) (DeviceFloatKernels.mapTanh)
    /// A function for elementwise arc sin of device elements
    let mapArcSin cmpVal =
        fold1ArgMaths cmpVal (asin) (DeviceFloatKernels.mapArcSin)
    /// A function for elementwise arc cos of device elements
    let mapArcCos cmpVal =
        fold1ArgMaths cmpVal (acos) (DeviceFloatKernels.mapArcCos)
    /// A function for elementwise arc tan of device elements
    let mapArcTan cmpVal =
        fold1ArgMaths cmpVal (atan) (DeviceFloatKernels.mapArcTan)
    /// A function for elementwise log_e of device elements
    let mapLog cmpVal =
        fold1ArgMaths cmpVal (log) (DeviceFloatKernels.mapLog)
    /// A function for elementwise log_10 of device elements
    let mapLog10 cmpVal =
        fold1ArgMaths cmpVal (log10) (DeviceFloatKernels.mapLog10)

    // Comparison

    /// A function for elementwise greater than comparison of device elements
    let mapGreaterThan cmpVal1 cmpVal2 =
        match (cmpVal1, cmpVal2) with 
        |ResComputeFloat d1, ResComputeFloat d2 -> ResComputeBool(d1 > d2)
        |ResComputeArray arr, ResComputeFloat d -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapGreaterThan) d arr (ResComputeBool(false)))
        |ResComputeFloat d, ResComputeArray arr -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapGreaterThan2) d arr (ResComputeBool(false)))
        |ResComputeArray arr1, ResComputeArray arr2 -> ResComputeArray(typeChangingMap2 (DeviceFloatKernels.map2GreaterThan) arr1 arr2 (ResComputeBool(false)))
        |_ -> raise <| System.NotSupportedException()

    /// A function for elementwise greater than or equal comparison of device elements
    let mapGreaterThanOrEqual cmpVal1 cmpVal2 =
        match (cmpVal1, cmpVal2) with 
        |ResComputeFloat d1, ResComputeFloat d2 -> ResComputeBool(d1 >= d2)
        |ResComputeArray arr, ResComputeFloat d -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapGreaterThanOrEqual) d arr (ResComputeBool(false)))
        |ResComputeFloat d, ResComputeArray arr -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapGreaterThanOrEqual2) d arr (ResComputeBool(false)))
        |ResComputeArray arr1, ResComputeArray arr2 -> ResComputeArray(typeChangingMap2 (DeviceFloatKernels.map2GreaterThanOrEqual) arr1 arr2 (ResComputeBool(false)))
        |_ -> raise <| System.NotSupportedException()

    /// A function for elementwise less than comparison of device elements
    let mapLessThan cmpVal1 cmpVal2 =
        match (cmpVal1, cmpVal2) with 
        |ResComputeFloat d1, ResComputeFloat d2 -> ResComputeBool(d1 < d2)
        |ResComputeArray arr, ResComputeFloat d -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapLessThan) d arr (ResComputeBool(false)))
        |ResComputeFloat d, ResComputeArray arr -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapLessThan2) d arr (ResComputeBool(false)))
        |ResComputeArray arr1, ResComputeArray arr2 -> ResComputeArray(typeChangingMap2 (DeviceFloatKernels.map2LessThan) arr1 arr2 (ResComputeBool(false)))
        |_ -> raise <| System.InvalidOperationException()

    /// A function for elementwise less than or equal comparison of device elements
    let mapLessThanOrEqual cmpVal1 cmpVal2 =
        match (cmpVal1, cmpVal2) with 
        |ResComputeFloat d1, ResComputeFloat d2 -> ResComputeBool(d1 <= d2)
        |ResComputeArray arr, ResComputeFloat d -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapLessThanOrEqual) d arr (ResComputeBool(false)))
        |ResComputeFloat d, ResComputeArray arr -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapLessThanOrEqual2) d arr (ResComputeBool(false)))
        |ResComputeArray arr1, ResComputeArray arr2 -> ResComputeArray(typeChangingMap2 (DeviceFloatKernels.map2LessThanOrEqual) arr1 arr2 (ResComputeBool(false)))
        |_ -> raise <| System.NotSupportedException()

    // Equality

    /// A function for elementwise equality checking of device elements
    let mapEquality cmpVal1 cmpVal2 =
        match (cmpVal1, cmpVal2) with 
        |ResComputeFloat d1, ResComputeFloat d2 -> ResComputeBool(d1 = d2)
        |ResComputeBool bl1, ResComputeBool bl2 -> ResComputeBool(bl1 = bl2)
        |ResComputeFloat d, ResComputeArray arr -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapEquality) d arr (ResComputeBool(false)))
        |ResComputeArray arr, ResComputeFloat d -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapEquality) d arr (ResComputeBool(false)))
        |ResComputeArray arr1, ResComputeArray arr2 -> ResComputeArray(typeChangingMap2 (DeviceFloatKernels.map2Equality) arr1 arr2 (ResComputeBool(false)))
        |_ -> raise <| System.NotSupportedException()

    /// A function for elementwise inequality checking of device elements
    let mapInequality cmpVal1 cmpVal2 =
        match (cmpVal1, cmpVal2) with 
        |ResComputeFloat d1, ResComputeFloat d2 -> ResComputeBool(d1 <> d2)
        |ResComputeBool bl1, ResComputeBool bl2 -> ResComputeBool(bl1 <> bl2)
        |ResComputeFloat d, ResComputeArray arr -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapInequality) d arr (ResComputeBool(false)))
        |ResComputeArray arr, ResComputeFloat d -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapInequality) d arr (ResComputeBool(false)))
        |ResComputeArray arr1, ResComputeArray arr2 -> ResComputeArray(typeChangingMap2 (DeviceFloatKernels.map2NotEquality) arr1 arr2 (ResComputeBool(false)))
        |_ -> raise <| System.NotSupportedException()

    // Conditional

    /// A function for elementwise conditional AND of device elements
    let mapConditionalAnd cmpVal1 cmpVal2 =
        foldCommutativeConditional cmpVal1 cmpVal2
            ( && ) (DeviceBoolKernels.mapConditionalAnd) (DeviceBoolKernels.map2ConditionalAnd)

    /// A function for elementwise conditional OR of device elements
    let mapConditionalOr cmpVal1 cmpVal2 =
        foldCommutativeConditional cmpVal1 cmpVal2
            ( || ) (DeviceBoolKernels.mapConditionalOr) (DeviceBoolKernels.map2ConditionalOr)

