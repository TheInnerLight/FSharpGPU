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

module FloatHelper = 
    let incrFloatBySmallest x = 
        let lng = System.BitConverter.DoubleToInt64Bits(x)
        match x with 
        |x when x > 0.0 -> System.BitConverter.Int64BitsToDouble(lng + 1L)
        |0.0 -> System.Double.Epsilon
        |_ -> System.BitConverter.Int64BitsToDouble(lng - 1L)
    let decrFloatBySmallest x = 
        let lng = System.BitConverter.DoubleToInt64Bits(x)
        match x with 
        |x when x > 0.0 -> System.BitConverter.Int64BitsToDouble(lng - 1L)
        |0.0 -> System.Double.Epsilon
        |_ -> System.BitConverter.Int64BitsToDouble(lng + 1L)

[<TestClass>]
type FloatOperatorEqualityUnitTests() = 
    let tolerance = 1e-20

    /// Unit tests for equality operator
    [<TestMethod>]
    member x.EqualityTests () = 
        let rnd = System.Random()
        // Test1
        let array = Array.init (1000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = (cudaArray, cudaArray) ||> DeviceArray.map2 ( fun x y -> x .=. y ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult |> Array.forall id)
        // Test2
        let array = Array.init (1) (fun i -> 82873963.2292410628)
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.map ( fun x -> x .=. 82873963.2292410628 ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult.[0])
        // Test3
        let array = Array.init (1) (fun i -> -6541.791131529)
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.map ( fun x -> x .=. -6541.791131529 ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult.[0])
        // Test4
        let array1 = Array.init (1000) (fun i -> sin <| float i)
        let array2 = Array.init (1000) (fun i -> sin <| float i)
        let cudaArray1 = DeviceArray.ofArray array1
        let cudaArray2 = DeviceArray.ofArray array2
        let cudaResult = (cudaArray1, cudaArray2) ||> DeviceArray.map2 ( fun x y -> x .=. y ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult |> Array.forall id)
        // Test5
        let array = Array.init (1000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = (cudaArray, cudaArray) ||> DeviceArray.map2 ( fun x y -> (x+1.0) .<>. y ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult |> Array.forall id)
    /// Unit tests for inequality operator
    [<TestMethod>]
    member x.InequalityTests () = 
        let rnd = System.Random()
        // Test1
        let array = Array.init (1000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = (cudaArray, cudaArray) ||> DeviceArray.map2 ( fun x y -> (x+1.0) .<>. y ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult |> Array.forall id)
        // Test2
        let array = Array.init (1) (fun i -> 763445.5508367985)
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.map ( fun x -> x .<>. 79.674689 ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult.[0])
        // Test3
        let array = Array.init (1) (fun i -> -35390.791131529)
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.map ( fun x -> x .<>. -24591.004533 ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult.[0])
        // Test4
        let array = Array.init (1) (fun i -> 95875.4577050924)
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.map ( fun x -> x .<>. -7050924.2129105 ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult.[0])
        // Test5
        let array = Array.init (1) (fun i -> -858.791131529)
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.map ( fun x -> x .<>. 103282.77614 ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult.[0])
[<TestClass>]
type FloatOperatorComparisonUnitTests() = 
    let tolerance = 1e-20
    /// Unit tests for greater than operator
    [<TestMethod>]
    member x.GreaterThanTests () = 
        let rnd = System.Random()
        
        // Test1
        let array = Array.init (1000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let array2 = array |> Array.map (fun x ->  FloatHelper.decrFloatBySmallest x )
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> x .>. y ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult |> Array.forall id)
        // Test2
        let array = Array.init (1000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let array2 = array |> Array.map (fun x -> FloatHelper.incrFloatBySmallest x) 
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> x .>. y ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult |> Array.forall not)
        // Test3
        let array = Array.init (1000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.map ( fun x -> x .>. -1.0 ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult |> Array.forall id)
    /// Unit tests for greater than or equal operator
    [<TestMethod>]
    member x.GreaterThanOrEqualTests () = 
        let rnd = System.Random()
        // Test1
        let array = Array.init (1000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let array2 = array |> Array.map (fun x ->  FloatHelper.decrFloatBySmallest x )
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> x .>=. y ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult |> Array.forall id)
        // Test2
        let array = Array.init (1000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let array2 = array |> Array.map (fun x -> FloatHelper.incrFloatBySmallest x) 
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> x .>=. y ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult |> Array.forall not)
        // Test3
        let array = Array.init (1000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = (cudaArray, cudaArray) ||> DeviceArray.map2 ( fun x y -> x .>=. y ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult |> Array.forall id)
        // Test4
        let array = Array.init (1000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.map ( fun x -> x .>=. 0.0 ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult |> Array.forall id)
    [<TestMethod>]
    member x.LessThanTests () = 
        let rnd = System.Random()
        // Test1
        let array = Array.init (1000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let array2 = array |> Array.map (fun x ->  FloatHelper.incrFloatBySmallest x )
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> x .<. y ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult |> Array.forall id)
        // Test2
        let array = Array.init (1000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let array2 = array |> Array.map (fun x -> FloatHelper.decrFloatBySmallest x) 
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> x .<. y ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult |> Array.forall not)
        // Test3
        let array = Array.init (1000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.map ( fun x -> x .<. 2.0 ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult |> Array.forall id)
    /// Unit tests for greater than or equal operator
    [<TestMethod>]
    member x.LessThanOrEqualTests () = 
        let rnd = System.Random()
        // Test1
        let array = Array.init (1000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let array2 = array |> Array.map (fun x ->  FloatHelper.incrFloatBySmallest x )
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> x .<=. y ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult |> Array.forall id)
        // Test2
        let array = Array.init (1000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let array2 = array |> Array.map (fun x -> FloatHelper.decrFloatBySmallest x) 
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> x .<=. y ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult |> Array.forall not)
        // Test3
        let array = Array.init (1000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = (cudaArray, cudaArray) ||> DeviceArray.map2 ( fun x y -> x .<=. y ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult |> Array.forall id)
        // Test4
        let array = Array.init (1000) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = cudaArray |> DeviceArray.map ( fun x -> x .<=. 1.0 ) |> Array.ofDeviceArray
        Assert.AreEqual(true, cudaResult |> Array.forall id)
[<TestClass>]
type FloatOperatorArithmeticUnitTests() = 
    let tolerance = 1e-20
    /// Unit tests for addition operator
    [<TestMethod>]
    member x.AdditionTests () = 
        let rnd = System.Random()
        // Test1
        let array = Array.init (10000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let array2 = Array.init (10000) (fun i -> rnd.NextDouble()  * 1e19 - rnd.NextDouble() * 1e19)
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult = (array, array2) ||> Array.map2 ( fun x y -> x + y )
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 (fun x y  -> x + y) |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // Test2
        let constant = rnd.NextDouble()
        let array = Array.init (10000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let cudaArray = DeviceArray.ofArray array
        let cpuResult = array |> Array.map ( fun x -> x + constant )
        let cudaResult = cudaArray |> DeviceArray.map (fun x  -> x + constant) |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // Test3
        let constant = rnd.NextDouble()
        let array = Array.init (10000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let cudaArray = DeviceArray.ofArray array
        let cpuResult = array |> Array.map ( fun x -> x + constant )
        let cudaResult = cudaArray |> DeviceArray.map (fun x  -> x + constant) |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
    /// Unit tests for subtraction operator
    [<TestMethod>]
    member x.SubtractionTests () = 
        let rnd = System.Random()
        // Test1
        let array = Array.init (10000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let array2 = Array.init (10000) (fun i -> rnd.NextDouble()  * 1e19 - rnd.NextDouble() * 1e19)
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult = (array, array2) ||> Array.map2 ( fun x y -> x - y )
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 (fun x y  -> x - y) |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // Test2
        let array = Array.init (10000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let array2 = Array.init (10000) (fun i -> rnd.NextDouble()  * 1e19 - rnd.NextDouble() * 1e19)
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult = (array, array2) ||> Array.map2 ( fun x y -> y - x )
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 (fun x y  -> y - x) |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // Test3
        let constant = rnd.NextDouble()
        let array = Array.init (10000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let cudaArray = DeviceArray.ofArray array
        let cpuResult = array |> Array.map ( fun x -> x - constant )
        let cudaResult = cudaArray |> DeviceArray.map (fun x  -> x - constant) |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // Test4
        let constant = rnd.NextDouble()
        let array = Array.init (10000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let cudaArray = DeviceArray.ofArray array
        let cpuResult = array |> Array.map ( fun x -> constant - x )
        let cudaResult = cudaArray |> DeviceArray.map (fun x  -> constant - x) |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
    /// Unit tests for multiplication operator
    [<TestMethod>]
    member x.MultiplicationTests () = 
        let rnd = System.Random()
        // Test1
        let array = Array.init (10000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let array2 = Array.init (10000) (fun i -> rnd.NextDouble()  * 1e19 - rnd.NextDouble() * 1e19)
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult = (array, array2) ||> Array.map2 ( fun x y -> x * y )
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 (fun x y  -> x * y) |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // Test2
        let constant = rnd.NextDouble()
        let array = Array.init (10000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let cudaArray = DeviceArray.ofArray array
        let cpuResult = array |> Array.map ( fun x -> x * constant )
        let cudaResult = cudaArray |> DeviceArray.map (fun x  -> x * constant) |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // Test3
        let constant = rnd.NextDouble()
        let array = Array.init (10000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let cudaArray = DeviceArray.ofArray array
        let cpuResult = array |> Array.map ( fun x -> x * constant )
        let cudaResult = cudaArray |> DeviceArray.map (fun x  -> x * constant) |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
    /// Unit tests for division operator
    [<TestMethod>]
    member x.DivisionTests () = 
        let rnd = System.Random()
        // Test
        let array = Array.init (100000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let array2 = Array.init (100000) (fun i -> rnd.NextDouble()  * 1e19 - rnd.NextDouble() * 1e19)
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult = (array, array2) ||> Array.map2 ( fun x y -> x / y )
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 (fun x y  -> x / y) |> Array.ofDeviceArray
        let cpuResult2 = (array, array2) ||> Array.map2 ( fun x y -> y / x )
        let cudaResult2 = (cudaArray, cudaArray2) ||> DeviceArray.map2 (fun x y  -> y / x) |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        (cpuResult2, cudaResult2) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // Test2
        let constant = rnd.NextDouble()
        let array = Array.init (10000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let cudaArray = DeviceArray.ofArray array
        let cpuResult = array |> Array.map ( fun x -> x / constant )
        let cudaResult = cudaArray |> DeviceArray.map (fun x  -> x / constant) |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // Test3
        let constant = rnd.NextDouble()
        let array = Array.init (10000) (fun i -> rnd.NextDouble() * 1e19 - rnd.NextDouble() * 1e19)
        let cudaArray = DeviceArray.ofArray array
        let cpuResult = array |> Array.map ( fun x -> constant / x )
        let cudaResult = cudaArray |> DeviceArray.map (fun x  -> constant / x ) |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))