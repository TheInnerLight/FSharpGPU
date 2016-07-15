(*This file is part of FSharpGPU.

FSharpGPU is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*)

(* Copyright © 2015-2016 Philip Curzon *)

namespace NovelFS.FSharpGPU.UnitTests

open System
open NovelFS.FSharpGPU
open FsCheck
open FsCheck.Xunit


type ``Function Tests`` =

    [<Property>]
    static member ``Using let binding to float value and adding to array value produces same result as on host`` (array : float array, value : float) = 
        let cudaArray = DeviceArray.ofArray array
        let cpuResult1 = array |> Array.map (fun x -> x + value)
        let cudaResult1 = 
            cudaArray 
            |> DeviceArray.map (fun x -> 
                let y = x
                let v = value
                y + v ) 
            |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.forall2 (TestConstants.checkFloatInRange TestConstants.tolerance)

    [<Property>]
    static member ``Using let binding to float value and adding to sin of array value produces same result as on host`` (array : float array, value : float) = 
        let cudaArray = DeviceArray.ofArray array
        let cpuResult = array |> Array.map (fun x -> sin x + value)
        let cudaResult = 
            cudaArray 
            |> DeviceArray.map (fun x -> 
                let y = sin x
                let v = value
                y + v )
            |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.forall2 (TestConstants.checkFloatInRange TestConstants.tolerance)

    [<Property>]
    static member ``Using function tuple binding to access (x, y) in device array and map with cos x + sin y produces same result as on host`` (tArray : (float *float) array) = 
        let array, array2 = Array.unzip tArray 
        let tArray = Array.zip array array2
        let cudaArray, cudaArray2 = DeviceArray.ofArray array, DeviceArray.ofArray array2
        let tCudaArray = DeviceArray.zip cudaArray cudaArray
        let cpuResult = tArray |> Array.map (fun (x,y) -> cos x + sin y)
        let cudaResult = 
            tCudaArray 
            |> DeviceArray.map (fun (x,y) -> cos x + sin y) 
            |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.forall2 (TestConstants.checkFloatInRange TestConstants.tolerance)

    [<Property>]
    static member ``Using function tuple binding to access (x, y) and returning (cos x, sin y) produces same result as on host`` (tArray : (float *float) array) = 
        let array, array2 = Array.unzip tArray
        let tArray = Array.zip array array2
        let cudaArray, cudaArray2 = DeviceArray.ofArray array, DeviceArray.ofArray array2
        let tCudaArray = DeviceArray.zip cudaArray cudaArray
        let cpuResultp1, cpuResultp2 = tArray |> Array.map (fun (x,y) -> cos x, sin y) |> Array.unzip
        let cudaResultp1, cudaResultp2 = 
            tCudaArray 
            |> DeviceArray.map (fun (x,y) -> cos x, sin y)
            |> DeviceArray.unzip
            |> fun (p1, p2) -> Array.ofDeviceArray p1, Array.ofDeviceArray p2
        (cpuResultp1, cudaResultp1) ||> Array.forall2 (TestConstants.checkFloatInRange TestConstants.tolerance)
        && (cpuResultp2, cudaResultp2) ||> Array.forall2 (TestConstants.checkFloatInRange TestConstants.tolerance)