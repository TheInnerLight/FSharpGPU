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

namespace NovelFS.FSharpGPU.UnitTests

open System
open NovelFS.FSharpGPU
open Microsoft.VisualStudio.TestTools.UnitTesting

[<TestClass>]
type DeviceArrayTupleUnitTests() = 
    /// Unit test for tuple map
    [<TestMethod>]
    member x.``tuple map two arrays`` () = 
        let rnd = Random()
        let tolerance = 1e-9
        // test 1
        let array = Array.init (10000) (fun i -> rnd.NextDouble())
        let tArray = Array.zip array array
        let cudaArray = DeviceArray.ofArray array
        let tCudaArray = DeviceArray.zip cudaArray cudaArray
        let cpuResultp1, cpuResultp2 = tArray |> Array.map (fun (x,y) -> cos x, sin y) |> Array.unzip
        let cudaResultp1, cudaResultp2 = 
            tCudaArray 
            |> DeviceArray.map (fun (x,y) -> cos x, sin y)
            |> DeviceArray.unzip
            |> fun (p1, p2) -> Array.ofDeviceArray p1, Array.ofDeviceArray p2
        (cpuResultp1, cudaResultp1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        (cpuResultp2, cudaResultp2) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))

    /// Unit test for tuple filter
    [<TestMethod>]
    member x.``tuple filter`` () = 
        let rnd = Random()
        let tolerance = 1e-9
        // test 1
        let array1 = Array.init (10000) (fun i -> rnd.NextDouble())
        let array2 = Array.init (10000) (fun i -> rnd.NextDouble())
        let tArray = Array.zip array1 array2
        let cudaArray1 = DeviceArray.ofArray array1
        let cudaArray2 = DeviceArray.ofArray array2
        let tCudaArray = DeviceArray.zip cudaArray1 cudaArray2
        let cpuResultp1, cpuResultp2 = tArray |> Array.map (fun (x,y) -> cos x, sin y) |> Array.filter (fun (x,y) -> x > 0.1 && y < 0.3) |> Array.unzip
        let cudaResultp1, cudaResultp2 = 
            tCudaArray 
            |> DeviceArray.map (fun (x,y) -> cos x, sin y)
            |> DeviceArray.filter (fun (x,y) -> (x .>. 0.1) .&&. ( y .<. 0.3))
            |> DeviceArray.unzip
            |> fun (p1, p2) -> Array.ofDeviceArray p1, Array.ofDeviceArray p2
        (cpuResultp1, cudaResultp1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        (cpuResultp2, cudaResultp2) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))

