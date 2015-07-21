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

exception CudaOutOfMemoryException of string

module internal DeviceInterop =
    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="createCUDAArray", CallingConvention = CallingConvention.Cdecl)>]
    extern int createUninitialisedArray(int size, int typeSize, IntPtr& handle)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="createCUDADoubleArray", CallingConvention = CallingConvention.Cdecl)>]
    extern int createUninitialisedCUDADoubleArray(int size, IntPtr& handle)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="initialiseCUDADoubleArray", CallingConvention = CallingConvention.Cdecl)>]
    extern int initialiseCUDADoubleArray(double[] flts, int n, IntPtr& handle)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="retrieveCUDADoubleArray", CallingConvention = CallingConvention.Cdecl)>]
    extern int retrieveCUDADoubleArray(IntPtr handle, int offset, double[] flts, int n)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="createCUDABoolArray", CallingConvention = CallingConvention.Cdecl)>]
    extern int createUninitialisedCUDABoolArray(int size, IntPtr& handle)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="initialiseCUDABoolArray", CallingConvention = CallingConvention.Cdecl)>]
    extern int initialiseCUDABoolArray(bool[] flts, int n, IntPtr& handle)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="retrieveCUDABoolArray", CallingConvention = CallingConvention.Cdecl)>]
    extern int retrieveCUDABoolArray(IntPtr handle, int offset, int[] flts, int n)

    let cudaCallWithExceptionCheck func =
        let value = func
        match value with
        |0 -> ()
        |1 -> failwith "cudaErrorMissingConfiguration : The device function being invoked (usually via cudaLaunch()) was not previously configured via the cudaConfigureCall() function."
        |2 -> raise <| CudaOutOfMemoryException "The API call failed because it was unable to allocate enough memory to perform the requested operation."
        |3 -> failwith "cudaErrorInitializationError : The API call failed because the CUDA driver and runtime could not be initialized."
        |4 -> failwith "cudaErrorLaunchFailure : An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory."
        |5 -> failwith "cudaErrorPriorLaunchFailure : This indicated that a previous kernel launch failed. This was previously used for device emulation of kernel launches."
        |6 -> failwith "cudaErrorLaunchTimeout : This indicates that the device kernel took too long to execute."
        |7 -> failwith "cudaErrorLaunchOutOfResources : This indicates that a launch did not occur because it did not have appropriate resources. This error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count."
        |8 -> failwith "cudaErrorInvalidDeviceFunction : The requested device function does not exist or is not compiled for the proper device architecture."
        |9 -> failwith "cudaErrorInvalidConfiguration :	This indicates that a kernel launch is requesting resources that can never be satisfied by the current device. Requesting more shared memory per block than the device supports will trigger this error, as will requesting too many threads or blocks."
        |10 -> failwith "cudaErrorInvalidDevice : This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device."
        |11 -> failwith "cudaErrorInvalidValue : This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values."
        |_ -> failwith (sprintf "cuda error code %i" value)

module internal DeviceFloatKernels = 

    // Float to Float mappings

    // Addition

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapAdd", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapAdd(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmap2Add", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2Add(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    // Subtraction

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapSubtract", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapSubtract(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapSubtract2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapSubtract2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmap2Subtract", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2Subtract(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    // Multiplication

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapMultiply", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapMultiply(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmap2Multiply", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2Multiply(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    // Division

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapDivide", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapDivide(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapDivide2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapDivide2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmap2Divide", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2Divide(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    // Power

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapPower", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapPower(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapPower2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapPower2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmap2Power", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2Power(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    // Various maths functions

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapSqrt", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapSqrt(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapArcCos", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapArcCos(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapCos", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapCos(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapCosh", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapCosh(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapArcSin", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapArcSin(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapSin", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapSin(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapSinh", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapSinh(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapArcTan", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapArcTan(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapTan", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapTan(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapTanh", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapTanh(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapLog", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapLog(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddmapLog10", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapLog10(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

    // Float to Bool mappings

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="dbmapGT", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapGreaterThan(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="dbmapGT2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapGreaterThan2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="dbmap2GT", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2GreaterThan(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="dbmapGTE", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapGreaterThanOrEqual(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="dbmapGTE2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapGreaterThanOrEqual2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="dbmap2GTE", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2GreaterThanOrEqual(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="dbmapLT", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapLessThan(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="dbmapLT2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapLessThan2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="dbmap2LT", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2LessThan(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="dbmapLTE", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapLessThanOrEqual(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="dbmapLTE2", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapLessThanOrEqual2(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="dbmap2LTE", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2LessThanOrEqual(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="dbmapEquality", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapEquality(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="dbmap2Equality", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2Equality(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="dbmapNotEquality", CallingConvention = CallingConvention.Cdecl)>]
    extern int mapInequality(IntPtr inArr, int inputOffset, int inputN, double flt, IntPtr outArr)

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="dbmap2NotEquality", CallingConvention = CallingConvention.Cdecl)>]
    extern int map2NotEquality(IntPtr inArr1, int inOff1, IntPtr inArr2, int inOff2, int n, IntPtr outArr)

    // Float to Float reductions

    [<DllImport(@"..\..\..\Debug\FSharpGPUInterop.dll", EntryPoint="ddreduceToHalf", CallingConvention = CallingConvention.Cdecl)>]
    extern int reduceToHalf(IntPtr inArr, int inputOffset, int inputN, IntPtr outArr)

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
    /// A helper function for creating non-commutative arithmetic functions which can operate on many device types
    let private mapNonCommutativeArithmetic cmpVal1 cmpVal2 
        opFltVV opFltAV opFltVA opFltAA = // float value & float value, array & float value, float value & array, array & array
            match (cmpVal1, cmpVal2) with 
            |ResComputeFloat d1, ResComputeFloat d2 -> ResComputeFloat(opFltVV d1 d2)
            |ResComputeFloat d, ResComputeArray arr -> ResComputeArray(typePreservingMapWithConst (opFltVA) d arr)
            |ResComputeArray arr, ResComputeFloat d -> ResComputeArray(typePreservingMapWithConst (opFltAV) d arr)
            |ResComputeArray arr1, ResComputeArray arr2 -> ResComputeArray(typePreservingMap2 (opFltAA) arr1 arr2)
            |_ -> raise <| System.NotSupportedException()
    /// A helper function for creating commutative arithmetic functions which can operate on many device types
    let private mapCommutativeArithmetic cmpVal1 cmpVal2 opFltVV opFltAV opFltAA =
        mapNonCommutativeArithmetic cmpVal1 cmpVal2 
            opFltVV opFltAV opFltAV opFltAA // Float Operations

    // Arithmetic
    // ----------

    /// A function for elementwise addition of device elements
    let mapAdd cmpVal1 cmpVal2 =
        mapCommutativeArithmetic cmpVal1 cmpVal2 
            ( + ) (DeviceFloatKernels.mapAdd) (DeviceFloatKernels.map2Add) // Float Operations
    /// A function for elementwise subtraction of device elements
    let mapSubtract cmpVal1 cmpVal2 =
        mapNonCommutativeArithmetic cmpVal1 cmpVal2 
            ( - ) (DeviceFloatKernels.mapSubtract) (DeviceFloatKernels.mapSubtract2) (DeviceFloatKernels.map2Subtract) // Float Operations
    /// A function for elementwise multiplication of device elements
    let mapMultiply cmpVal1 cmpVal2 =
        mapCommutativeArithmetic cmpVal1 cmpVal2 
            ( * ) (DeviceFloatKernels.mapMultiply) (DeviceFloatKernels.map2Multiply) // Float Operations
    /// A function for elementwise division of device elements
    let mapDivide cmpVal1 cmpVal2 =
        mapNonCommutativeArithmetic cmpVal1 cmpVal2 
            ( / ) (DeviceFloatKernels.mapDivide) (DeviceFloatKernels.mapDivide2) (DeviceFloatKernels.map2Divide) // Float Operations
    /// A function for elementwise power raising of device elements
    let mapPower cmpVal1 cmpVal2 =
        mapNonCommutativeArithmetic cmpVal1 cmpVal2 
            ( ** ) (DeviceFloatKernels.mapPower) (DeviceFloatKernels.mapPower2) (DeviceFloatKernels.map2Power) // Float Operations

    // Maths functions
    // ---------------

    let mapSqrt cmpVal =
        match cmpVal with
        |ResComputeFloat d -> ResComputeFloat(sqrt d)
        |ResComputeArray arr -> ResComputeArray(typePreservingMap (DeviceFloatKernels.mapSqrt) arr)
        |_ -> raise <| System.NotSupportedException()

    let mapSin cmpVal =
        match cmpVal with
        |ResComputeFloat d -> ResComputeFloat(sin d)
        |ResComputeArray arr -> ResComputeArray(typePreservingMap (DeviceFloatKernels.mapSin) arr)
        |_ -> raise <| System.NotSupportedException()

    let mapCos cmpVal =
        match cmpVal with
        |ResComputeFloat d -> ResComputeFloat(cos d)
        |ResComputeArray arr -> ResComputeArray(typePreservingMap (DeviceFloatKernels.mapCos) arr)
        |_ -> raise <| System.NotSupportedException()

    let mapTan cmpVal =
        match cmpVal with
        |ResComputeFloat d -> ResComputeFloat(tan d)
        |ResComputeArray arr -> ResComputeArray(typePreservingMap (DeviceFloatKernels.mapTan) arr)
        |_ -> raise <| System.NotSupportedException()

    let mapSinh cmpVal =
        match cmpVal with
        |ResComputeFloat d -> ResComputeFloat(sinh d)
        |ResComputeArray arr -> ResComputeArray(typePreservingMap (DeviceFloatKernels.mapSinh) arr)
        |_ -> raise <| System.NotSupportedException()

    let mapCosh cmpVal =
        match cmpVal with
        |ResComputeFloat d -> ResComputeFloat(cosh d)
        |ResComputeArray arr -> ResComputeArray(typePreservingMap (DeviceFloatKernels.mapCosh) arr)
        |_ -> raise <| System.NotSupportedException()

    let mapTanh cmpVal =
        match cmpVal with
        |ResComputeFloat d -> ResComputeFloat(tanh d)
        |ResComputeArray arr -> ResComputeArray(typePreservingMap (DeviceFloatKernels.mapTanh) arr)
        |_ -> raise <| System.NotSupportedException()

    let mapArcSin cmpVal =
        match cmpVal with
        |ResComputeFloat d -> ResComputeFloat(asin d)
        |ResComputeArray arr -> ResComputeArray(typePreservingMap (DeviceFloatKernels.mapArcSin) arr)
        |_ -> raise <| System.NotSupportedException()

    let mapArcCos cmpVal =
        match cmpVal with
        |ResComputeFloat d -> ResComputeFloat(acos d)
        |ResComputeArray arr -> ResComputeArray(typePreservingMap (DeviceFloatKernels.mapArcCos) arr)
        |_ -> raise <| System.NotSupportedException()

    let mapArcTan cmpVal =
        match cmpVal with
        |ResComputeFloat d -> ResComputeFloat(atan d)
        |ResComputeArray arr -> ResComputeArray(typePreservingMap (DeviceFloatKernels.mapArcTan) arr)
        |_ -> raise <| System.NotSupportedException()

    let mapLog cmpVal =
        match cmpVal with
        |ResComputeFloat d -> ResComputeFloat(log d)
        |ResComputeArray arr -> ResComputeArray(typePreservingMap (DeviceFloatKernels.mapLog) arr)
        |_ -> raise <| System.NotSupportedException()

    let mapLog10 cmpVal =
        match cmpVal with
        |ResComputeFloat d -> ResComputeFloat(log10 d)
        |ResComputeArray arr -> ResComputeArray(typePreservingMap (DeviceFloatKernels.mapLog10) arr)
        |_ -> raise <| System.NotSupportedException()

    // Comparison

    let mapGreaterThan cmpVal1 cmpVal2 =
        match (cmpVal1, cmpVal2) with 
        |ResComputeFloat d1, ResComputeFloat d2 -> ResComputeBool(d1 > d2)
        |ResComputeArray arr, ResComputeFloat d -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapGreaterThan) d arr (ResComputeBool(false)))
        |ResComputeFloat d, ResComputeArray arr -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapGreaterThan2) d arr (ResComputeBool(false)))
        |ResComputeArray arr1, ResComputeArray arr2 -> ResComputeArray(typeChangingMap2 (DeviceFloatKernels.map2GreaterThan) arr1 arr2 (ResComputeBool(false)))
        |_ -> raise <| System.NotSupportedException()

    let mapGreaterThanOrEqual cmpVal1 cmpVal2 =
        match (cmpVal1, cmpVal2) with 
        |ResComputeFloat d1, ResComputeFloat d2 -> ResComputeBool(d1 >= d2)
        |ResComputeArray arr, ResComputeFloat d -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapGreaterThanOrEqual) d arr (ResComputeBool(false)))
        |ResComputeFloat d, ResComputeArray arr -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapGreaterThanOrEqual2) d arr (ResComputeBool(false)))
        |ResComputeArray arr1, ResComputeArray arr2 -> ResComputeArray(typeChangingMap2 (DeviceFloatKernels.map2GreaterThanOrEqual) arr1 arr2 (ResComputeBool(false)))
        |_ -> raise <| System.NotSupportedException()

    let mapLessThan cmpVal1 cmpVal2 =
        match (cmpVal1, cmpVal2) with 
        |ResComputeFloat d1, ResComputeFloat d2 -> ResComputeBool(d1 < d2)
        |ResComputeArray arr, ResComputeFloat d -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapLessThan) d arr (ResComputeBool(false)))
        |ResComputeFloat d, ResComputeArray arr -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapLessThan2) d arr (ResComputeBool(false)))
        |ResComputeArray arr1, ResComputeArray arr2 -> ResComputeArray(typeChangingMap2 (DeviceFloatKernels.map2LessThan) arr1 arr2 (ResComputeBool(false)))
        |_ -> raise <| System.InvalidOperationException()

    let mapLessThanOrEqual cmpVal1 cmpVal2 =
        match (cmpVal1, cmpVal2) with 
        |ResComputeFloat d1, ResComputeFloat d2 -> ResComputeBool(d1 <= d2)
        |ResComputeArray arr, ResComputeFloat d -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapLessThanOrEqual) d arr (ResComputeBool(false)))
        |ResComputeFloat d, ResComputeArray arr -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapLessThanOrEqual2) d arr (ResComputeBool(false)))
        |ResComputeArray arr1, ResComputeArray arr2 -> ResComputeArray(typeChangingMap2 (DeviceFloatKernels.map2LessThanOrEqual) arr1 arr2 (ResComputeBool(false)))
        |_ -> raise <| System.NotSupportedException()

    // Equality

    let mapEquality cmpVal1 cmpVal2 =
        match (cmpVal1, cmpVal2) with 
        |ResComputeFloat d1, ResComputeFloat d2 -> ResComputeBool(d1 = d2)
        |ResComputeBool bl1, ResComputeBool bl2 -> ResComputeBool(bl1 = bl2)
        |ResComputeFloat d, ResComputeArray arr -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapEquality) d arr (ResComputeBool(false)))
        |ResComputeArray arr, ResComputeFloat d -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapEquality) d arr (ResComputeBool(false)))
        |ResComputeArray arr1, ResComputeArray arr2 -> ResComputeArray(typeChangingMap2 (DeviceFloatKernels.map2Equality) arr1 arr2 (ResComputeBool(false)))
        |_ -> raise <| System.NotSupportedException()

    let mapInequality cmpVal1 cmpVal2 =
        match (cmpVal1, cmpVal2) with 
        |ResComputeFloat d1, ResComputeFloat d2 -> ResComputeBool(d1 <> d2)
        |ResComputeBool bl1, ResComputeBool bl2 -> ResComputeBool(bl1 <> bl2)
        |ResComputeFloat d, ResComputeArray arr -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapInequality) d arr (ResComputeBool(false)))
        |ResComputeArray arr, ResComputeFloat d -> ResComputeArray(typeChangingMapWithConst (DeviceFloatKernels.mapInequality) d arr (ResComputeBool(false)))
        |ResComputeArray arr1, ResComputeArray arr2 -> ResComputeArray(typeChangingMap2 (DeviceFloatKernels.map2NotEquality) arr1 arr2 (ResComputeBool(false)))
        |_ -> raise <| System.NotSupportedException()
