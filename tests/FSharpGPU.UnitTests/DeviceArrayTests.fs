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

module TestConstants =
    let tolerance = 1e-9

    let checkFloatInRange tolerance f1 f2 =
        match f1, f2 with
        |f1, f2 when System.Double.IsNaN(f1) && System.Double.IsNaN(f2) -> true
        |f1, f2 when System.Double.IsNegativeInfinity(f1) && System.Double.IsNegativeInfinity(f2) -> true
        |f1, f2 when System.Double.IsPositiveInfinity(f1) && System.Double.IsPositiveInfinity(f2) -> true
        |f1, f2 -> abs (f1 - f2) < tolerance  

type ``DeviceArrayUnitTests``() = 
   
    [<Property>]
    static member ``Map with + 1 to random device array produces same result as on host`` (array : float array) = 
        let cudaArray = DeviceArray.ofArray array
        let cpuResult1 = array |> Array.map (fun x -> x + 1.0)
        let cudaResult1 = cudaArray |> DeviceArray.map ( fun x -> x + 1.0 ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.forall2 (TestConstants.checkFloatInRange TestConstants.tolerance)


    [<Property>]
    static member ``Map to a constant value produces same result on device and host`` (array : float array) = 
        let cudaArray = DeviceArray.ofArray array
        let cpuResult1 = array |> Array.map (fun x -> 1.0)
        let cudaResult1 = cudaArray |> DeviceArray.map ( fun x -> devicefloat 1.0 ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.forall2 (TestConstants.checkFloatInRange TestConstants.tolerance)

    [<Property>]
    static member  ``Map2 + between two random device arrays produces same result as on host`` (tArray : (float *float) array) = 
        let array, array2 = Array.unzip tArray
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult1 = (array, array2) ||> Array.map2 (fun x y -> x + y)
        let cudaResult1 = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> x + y ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.forall2 (TestConstants.checkFloatInRange TestConstants.tolerance)

    [<Property>]
    static member ``Filter less than constant of 0.9 produces same result as on host`` (array : float array) = 
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.filter (fun x -> x .>. 0.9) |> Array.ofDeviceArray
        let cpuResult = array |> Array.filter (fun x -> x > 0.9)
        (cpuResult, cudaResult) ||> Array.forall2 (TestConstants.checkFloatInRange TestConstants.tolerance)

    [<Property>]
    static member ``Partition less than constant of 0.9 produces same result as on host`` (array : float array) = 
        let cudaArray = DeviceArray.ofArray array
        let cudaTResult, cudaFResult = cudaArray |> DeviceArray.partition (fun x -> x .>. 0.9)
        let cpuTResult, cpuFResult = array |> Array.partition (fun x -> x > 0.9)
        let res1 = cudaTResult |> Array.ofDeviceArray
        let res2 = cudaFResult |> Array.ofDeviceArray
        (cpuTResult, cudaTResult |> Array.ofDeviceArray) ||> Array.forall2 (fun a1 a2 -> abs (a1 - a2) < TestConstants.tolerance)
        && (cpuFResult, cudaFResult |> Array.ofDeviceArray) ||> Array.forall2 (fun a1 a2 -> abs (a1 - a2) < TestConstants.tolerance)

    [<Property>]
    static member ``Sum by x*2.0 produces same result as on host`` (array : float array) = 
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.sumBy (fun x -> x * 2.0) |> DeviceElement.toHost
        let cpuResult = array |> Array.sumBy (fun x -> x * 2.0)
        abs (cpuResult - cudaResult) < TestConstants.tolerance

    [<Property>]
    static member ``Reduce with + 6.0 * value produces same result as on host`` (array : float array) =  
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.associativeReduce (fun acc value -> acc + (6.0 * value)) |> DeviceElement.toHost
        let cpuResult = array |> Array.fold (fun acc value -> acc + (6.0 * value)) 0.0
        abs (cpuResult - cudaResult) < TestConstants.tolerance

