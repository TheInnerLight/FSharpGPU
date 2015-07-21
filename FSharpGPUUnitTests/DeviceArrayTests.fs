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
type DevieArrayUnitTests() = 
    /// Unit tests for DeviceArray.map
    [<TestMethod>]
    member x.MapTests () = 
        let rnd = Random()
        let tolerance = 1e-9
        // test 1
        let array = Array.init (100) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cpuResult1 = array |> Array.map (fun x -> x + 1.0)
        let cudaResult1 = cudaArray |> DeviceArray.map ( fun x -> x + 1.0 ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // test 2
        let array = Array.init (100) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cpuResult1 = array |> Array.map (fun x -> x - 19.0)
        let cudaResult1 = cudaArray |> DeviceArray.map ( fun x -> x - 19.0 ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // test 3
        let array = Array.init (100) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cpuResult1 = array |> Array.map (fun x -> sin x - 19.0 * x )
        let cudaResult1 = cudaArray |> DeviceArray.map ( fun x -> sin x - 19.0 * x ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // test 4
        let array = Array.init (100) (fun i -> rnd.NextDouble() * 10.0)
        let cudaArray = DeviceArray.ofArray array
        let cpuResult1 = array |> Array.map (fun x -> sqrt(x) / x /5.0 + 1.0/7.78)
        let cudaResult1 = cudaArray |> DeviceArray.map ( fun x -> (sqrt(x) / x /5.0 + 1.0/7.78) ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // test 5
        let array = Array.init (100) (fun i -> rnd.NextDouble() * 1.0e116)
        let cudaArray = DeviceArray.ofArray array
        let cpuResult1 = array |> Array.map (fun x -> x ** 5.731 *  x + 63784.0 - 938724.4)
        let cudaResult1 = cudaArray |> DeviceArray.map ( fun x -> x ** 5.731 *  x + 63784.0 - 938724.4 ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))

    [<TestMethod>]
    member x.Map2Tests () = 
        let rnd = Random()
        let tolerance = 1e-9
        //test 1
        let array = Array.init (100) (fun i -> rnd.NextDouble())
        let array2 = Array.init (100) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult1 = (array, array2) ||> Array.map2 (fun x y -> x + y)
        let cudaResult1 = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> x + y ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        //test 2
        let array = Array.init (100) (fun i -> rnd.NextDouble())
        let array2 = Array.init (100) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult1 = (array, array2) ||> Array.map2 (fun x y -> x - y)
        let cudaResult1 = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> x - y ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        //test 3
        let array = Array.init (100) (fun i -> rnd.NextDouble())
        let array2 = Array.init (100) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult1 = (array, array2) ||> Array.map2 (fun x y -> sin x - cos y)
        let cudaResult1 = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> sin x - cos y ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        //test 4
        let array = Array.init (100) (fun i -> rnd.NextDouble())
        let array2 = Array.init (100) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult1 = (array, array2) ||> Array.map2 (fun x y -> sqrt x -  y ** 11.27)
        let cudaResult1 = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> sqrt x - y ** 11.27 ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))