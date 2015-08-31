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
        let array = Array.init (10000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cpuResult1 = array |> Array.map (fun x -> x + 1.0)
        let cudaResult1 = cudaArray |> DeviceArray.map ( fun x -> x + 1.0 ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // test 2
        let array = Array.init (10000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cpuResult1 = array |> Array.map (fun x -> x - 19.0)
        let cudaResult1 = cudaArray |> DeviceArray.map ( fun x -> x - 19.0 ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // test 3
        let array = Array.init (10000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cpuResult1 = array |> Array.map (fun x -> sin x - 19.0 * x )
        let cudaResult1 = cudaArray |> DeviceArray.map ( fun x -> sin x - 19.0 * x ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // test 4
        let array = Array.init (10000) (fun i -> rnd.NextDouble() * 10.0)
        let cudaArray = DeviceArray.ofArray array
        let cpuResult1 = array |> Array.map (fun x -> sqrt(x) / x /5.0 + 1.0/7.78)
        let cudaResult1 = cudaArray |> DeviceArray.map ( fun x -> (sqrt(x) / x /5.0 + 1.0/7.78) ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // test 5
        let array = Array.init (10000) (fun i -> rnd.NextDouble() * 1.0e116)
        let cudaArray = DeviceArray.ofArray array
        let cpuResult1 = array |> Array.map (fun x -> x ** 5.731 *  x + 63784.0 - 938724.4)
        let cudaResult1 = cudaArray |> DeviceArray.map ( fun x -> x ** 5.731 *  x + 63784.0 - 938724.4 ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))

    [<TestMethod>]
    member x.Map2Tests () = 
        let rnd = Random()
        let tolerance = 1e-9
        //test 1
        let array = Array.init (10000) (fun i -> rnd.NextDouble())
        let array2 = Array.init (10000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult1 = (array, array2) ||> Array.map2 (fun x y -> x + y)
        let cudaResult1 = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> x + y ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        //test 2
        let array = Array.init (10000) (fun i -> rnd.NextDouble())
        let array2 = Array.init (10000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult1 = (array, array2) ||> Array.map2 (fun x y -> x - y)
        let cudaResult1 = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> x - y ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        //test 3
        let array = Array.init (10000) (fun i -> rnd.NextDouble())
        let array2 = Array.init (10000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult1 = (array, array2) ||> Array.map2 (fun x y -> sin x - cos y)
        let cudaResult1 = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> sin x - cos y ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        //test 4
        let array = Array.init (10000) (fun i -> rnd.NextDouble())
        let array2 = Array.init (10000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult1 = (array, array2) ||> Array.map2 (fun x y -> sqrt x -  y ** 11.27)
        let cudaResult1 = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> sqrt x - y ** 11.27 ) |> Array.ofDeviceArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))

    [<TestMethod>]
    member x.FilterTests () = 
        let rnd = Random()
        let tolerance = 1e-9
        // test1
        let array = Array.init (10000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.filter (fun x -> x .>. 0.5) |> Array.ofDeviceArray
        let cpuResult = array |> Array.filter (fun x -> x > 0.5)
        (cpuResult, cudaResult) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // test2
        let array = Array.init (10000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.filter (fun x -> x .<. 0.5) |> Array.ofDeviceArray
        let cpuResult = array |> Array.filter (fun x -> x < 0.5)
        (cpuResult, cudaResult) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // test3
        let array = Array.init (10000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.filter (fun x -> x ** 2.0 .<. 0.5) |> Array.ofDeviceArray
        let cpuResult = array |> Array.filter (fun x -> x ** 2.0 < 0.5)
        (cpuResult, cudaResult) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // test4
        let array = Array.init (10000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.filter (fun x -> (x ** 2.0 .<. 0.25) .&&. (x .>. 0.1)) |> Array.ofDeviceArray
        let cpuResult = array |> Array.filter (fun x -> (x ** 2.0 < 0.25) && (x > 0.1))
        (cudaResult, cpuResult) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // test5
        let array = Array.init (10000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.filter (fun x -> (x ** 2.0 .<. 0.25) .||. ((x .>. 0.7) .&&. (x .<. 0.8))) |> Array.ofDeviceArray
        let cpuResult = array |> Array.filter (fun x -> (x ** 2.0 < 0.25) || ((x > 0.7) && (x < 0.8)))
        (cpuResult, cudaResult) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
    [<TestMethod>]
    member x.SummationTests () = 
        let rnd = Random()
        let tolerance = 1e-9
        // tests
        let array = Array.init (3571) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.sumBy (fun x -> x * 2.0)
        let cpuResult = array |> Array.sumBy (fun x -> x * 2.0)
        Assert.AreEqual(cpuResult, cudaResult, tolerance)
        let array = Array.init (3989) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.sumBy (fun x -> x + 3.0)
        let cpuResult = array |> Array.sumBy (fun x -> x + 3.0)
        Assert.AreEqual(cpuResult, cudaResult, tolerance)
        let array = Array.init (9876) (fun i -> float i * 2.0)
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.sumBy (fun x -> 1.0 / x )
        let cpuResult = array |> Array.sumBy (fun x -> 1.0 / x)
        Assert.AreEqual(cpuResult, cudaResult, tolerance)
        let array = Array.init (8752) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.sumBy (fun x -> sin x )
        let cpuResult = array |> Array.sumBy (fun x -> sin x)
        Assert.AreEqual(cpuResult, cudaResult, tolerance)
        let array = Array.init (4831) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.sumBy (fun x -> sqrt x )
        let cpuResult = array |> Array.sumBy (fun x -> sqrt x)
        Assert.AreEqual(cpuResult, cudaResult, tolerance)
        let array = Array.init (4096) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.sumBy (fun x -> x ** 0.4)
        let cpuResult = array |> Array.sumBy (fun x -> x ** 0.4)
        Assert.AreEqual(cpuResult, cudaResult, tolerance)

    [<TestMethod>]
    member x.ReductionTests () = 
        let rnd = Random()
        let tolerance = 1e-9
        // test 1
        let array = Array.init (10000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.associativeReduce (fun acc value -> acc + (6.0 * value))
        let cpuResult = array |> Array.reduce (fun acc value -> acc + (6.0 * value))
        Assert.AreEqual(cpuResult, cudaResult, tolerance)
        // test 2
        let array = Array.init (10000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.associativeReduce (fun acc x -> acc + (sin x))
        let cpuResult = array |> Array.reduce (fun acc x -> acc + (sin x))
        Assert.AreEqual(cpuResult, cudaResult, tolerance)
