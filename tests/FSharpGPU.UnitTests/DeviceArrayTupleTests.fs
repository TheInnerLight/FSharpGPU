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


type DeviceArrayTupleUnitTests = 
    /// Unit test for tuple map
    [<Property>]
    static member ``Map float tuple to cos x, sin y produces same value as on host`` (tArray : (float * float) array) = 
        let arr1, arr2 = Array.unzip tArray
        let cudaArray1, cudaArray2 = DeviceArray.ofArray arr1, DeviceArray.ofArray arr2
        let tCudaArray = DeviceArray.zip cudaArray1 cudaArray2
        let cpuResultp1, cpuResultp2 = tArray |> Array.map (fun (x,y) -> cos x, sin y) |> Array.unzip
        let cudaResultp1, cudaResultp2 = 
            tCudaArray 
            |> DeviceArray.map (fun (x,y) -> cos x, sin y)
            |> DeviceArray.unzip
            |> fun (p1, p2) -> Array.ofDeviceArray p1, Array.ofDeviceArray p2
        (cpuResultp1, cudaResultp1) ||> Array.forall2 (TestConstants.checkFloatInRange TestConstants.tolerance)
        && (cpuResultp2, cudaResultp2) ||> Array.forall2 (TestConstants.checkFloatInRange TestConstants.tolerance)

    /// Unit test for tuple filter
    [<Property>]
    static member ``Filter float tuple by mapping to cos x, sin y and filtering by x > 0.1 and y < 0.3 produces same value as on host`` (tArray : (float * float) array) = 
        let arr1, arr2 = Array.unzip tArray
        let cudaArray1, cudaArray2 = DeviceArray.ofArray arr1, DeviceArray.ofArray arr2
        let cudaArray1 = DeviceArray.ofArray arr1
        let cudaArray2 = DeviceArray.ofArray arr2
        let tCudaArray = DeviceArray.zip cudaArray1 cudaArray2
        let cpuResultp1, cpuResultp2 = tArray |> Array.map (fun (x,y) -> cos x, sin y) |> Array.filter (fun (x,y) -> x > 0.1 && y < 0.3) |> Array.unzip
        let cudaResultp1, cudaResultp2 = 
            tCudaArray 
            |> DeviceArray.map (fun (x,y) -> cos x, sin y)
            |> DeviceArray.filter (fun (x,y) -> (x .>. 0.1) .&&. ( y .<. 0.3))
            |> DeviceArray.unzip
            |> fun (p1, p2) -> Array.ofDeviceArray p1, Array.ofDeviceArray p2
        (cpuResultp1, cudaResultp1) ||> Array.forall2 (TestConstants.checkFloatInRange TestConstants.tolerance)
        && (cpuResultp2, cudaResultp2) ||> Array.forall2 (TestConstants.checkFloatInRange TestConstants.tolerance)

