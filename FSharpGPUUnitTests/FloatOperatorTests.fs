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

open NovelFS.FSharpGPU
open Microsoft.VisualStudio.TestTools.UnitTesting

[<TestClass>]
type FloatComparisonAndEqualityOperatorUnitTests() = 
    /// Unit tests for equality operator
    [<TestMethod>]
    member x.EqualityTests () = 
        let rnd = System.Random()
        // Test1
        let array = Array.init (1000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = (cudaArray, cudaArray) ||> DeviceArray.map2 <@ fun x y -> x .=. y @> |> Array.ofCudaArray
        Assert.AreEqual(true, cudaResult |> Array.forall id)
        // Test2
        let array = Array.init (1) (fun i -> 82873963.2292410628)
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.map <@ fun x -> x .=. 82873963.2292410628 @> |> Array.ofCudaArray
        Assert.AreEqual(true, cudaResult.[0])
        // Test3
        let array = Array.init (1) (fun i -> -6541.791131529)
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.map <@ fun x -> x .=. -6541.791131529 @> |> Array.ofCudaArray
        Assert.AreEqual(true, cudaResult.[0])
        // Test4
        let array1 = Array.init (1000) (fun i -> sin <| float i)
        let array2 = Array.init (1000) (fun i -> sin <| float i)
        let cudaArray1 = DeviceArray.ofArray array1
        let cudaArray2 = DeviceArray.ofArray array2
        let cudaResult = (cudaArray1, cudaArray2) ||> DeviceArray.map2 <@ fun x y -> x .=. y @> |> Array.ofCudaArray
        Assert.AreEqual(true, cudaResult |> Array.forall id)
        // Test5
        let array = Array.init (1000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = (cudaArray, cudaArray) ||> DeviceArray.map2 <@ fun x y -> (x+1.0) .<>. y @> |> Array.ofCudaArray
        Assert.AreEqual(true, cudaResult |> Array.forall id)
    /// Unit tests for inequality operator
    [<TestMethod>]
    member x.InequalityTests () = 
        let rnd = System.Random()
        // Test1
        let array = Array.init (1000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = (cudaArray, cudaArray) ||> DeviceArray.map2 <@ fun x y -> (x+1.0) .<>. y @> |> Array.ofCudaArray
        Assert.AreEqual(true, cudaResult |> Array.forall id)
        // Test2
        let array = Array.init (1) (fun i -> 763445.5508367985)
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.map <@ fun x -> x .<>. 79.674689 @> |> Array.ofCudaArray
        Assert.AreEqual(true, cudaResult.[0])
        // Test3
        let array = Array.init (1) (fun i -> -35390.791131529)
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.map <@ fun x -> x .<>. -24591.004533 @> |> Array.ofCudaArray
        Assert.AreEqual(true, cudaResult.[0])
        // Test4
        let array = Array.init (1) (fun i -> 95875.4577050924)
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.map <@ fun x -> x .<>. -7050924.2129105 @> |> Array.ofCudaArray
        Assert.AreEqual(true, cudaResult.[0])
        // Test5
        let array = Array.init (1) (fun i -> -858.791131529)
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.map <@ fun x -> x .<>. 103282.77614 @> |> Array.ofCudaArray
        Assert.AreEqual(true, cudaResult.[0])