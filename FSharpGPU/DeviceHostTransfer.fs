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

/// Functions for transfering arrays and elements between host and device
module DeviceHostTransfer =
    /// A helper type to support conversion between host and device elements
    type DeviceArrayCreator = 
        |DeviceArrayCreator

        /// transfer float array to device
        static member HostArrayToGpuArray (DeviceArrayCreator, arr : float[]) =
            let mutable cudaPtr = System.IntPtr(0)
            GeneralDevice.initialiseCUDADoubleArray(arr, Array.length arr, &cudaPtr) |> DeviceInterop.checkCudaResponse
            new devicearray<devicefloat>(new ComputeArray(ComputeDataType.ComputeFloat, cudaPtr, Array.length arr, FullArray, UserGenerated))

        /// transfer bool array to device
        static member HostArrayToGpuArray (DeviceArrayCreator, arr : bool[]) =
            let mutable cudaPtr = System.IntPtr(0)
            GeneralDevice.initialiseCUDABoolArray(arr, Array.length arr, &cudaPtr) |> DeviceInterop.checkCudaResponse
            new devicearray<devicefloat>(new ComputeArray(ComputeDataType.ComputeFloat, cudaPtr, Array.length arr, FullArray, UserGenerated))

        /// transfer device float array to host
        static member GpuArrayToHostArray (DeviceArrayCreator, array : devicearray<devicefloat>) =
            let devArray = ComputeResult.assumeSingleton (array.DeviceArrays)
            match devArray.ArrayType with
            |ComputeDataType.ComputeFloat ->
                let hostArray = Array.zeroCreate<float> (devArray.Length)
                GeneralDevice.retrieveCUDADoubleArray(devArray.CudaPtr, devArray.Offset, hostArray, hostArray.Length) |> DeviceInterop.checkCudaResponse
                hostArray
            |_ -> failwith "Invalid type"

        /// transfer device bool array to host
        static member GpuArrayToHostArray (DeviceArrayCreator, array : devicearray<devicebool>) =
            let devArray = ComputeResult.assumeSingleton (array.DeviceArrays)
            match devArray.ArrayType with
            |ComputeDataType.ComputeBool ->
                let hostArray = Array.zeroCreate<int> (devArray.Length)
                GeneralDevice.retrieveCUDABoolArray(devArray.CudaPtr, devArray.Offset, hostArray, hostArray.Length) |> DeviceInterop.checkCudaResponse
                hostArray |> Array.map (function |0 -> false; |_ -> true)
            |_ -> failwith "Invalid type"

        /// transfer device float element to host
        static member GpuElementToHostElement (DeviceArrayCreator, array : deviceelement<devicefloat>) =
            let devArray = array.DeviceArray
            match devArray.ArrayType with
            |ComputeDataType.ComputeFloat ->
                let hostArray = Array.zeroCreate<float> (devArray.Length)
                GeneralDevice.retrieveCUDADoubleArray(devArray.CudaPtr, devArray.Offset, hostArray, hostArray.Length) |> DeviceInterop.checkCudaResponse
                hostArray |> Array.head
            |_ -> failwith "Invalid type"

        /// transfer device bool element to host
        static member GpuElementToHostElement (DeviceArrayCreator, array : deviceelement<devicebool>) =
            let devArray = array.DeviceArray
            match devArray.ArrayType with
            |ComputeDataType.ComputeBool ->
                let hostArray = Array.zeroCreate<int> (devArray.Length)
                GeneralDevice.retrieveCUDABoolArray(devArray.CudaPtr, devArray.Offset, hostArray, hostArray.Length) |> DeviceInterop.checkCudaResponse
                hostArray |> Array.map (function |0 -> false; |_ -> true) |> Array.head
            |_ -> failwith "Invalid type"

    /// copy an array to the device
    let inline copyArrayToDevice arr =
        ((^T or ^U) : (static member HostArrayToGpuArray : ^T * ^U -> ^S) (DeviceArrayCreator, arr))

    /// copy an array to the device
    let inline copyArrayToHost arr =
        ((^T or ^U) : (static member GpuArrayToHostArray : ^T * ^U -> ^S) (DeviceArrayCreator, arr))

    /// copy an array to the device
    let inline copyElementToHost elem =
        ((^T or ^U) : (static member GpuElementToHostElement : ^T * ^U -> ^S) (DeviceArrayCreator, elem))