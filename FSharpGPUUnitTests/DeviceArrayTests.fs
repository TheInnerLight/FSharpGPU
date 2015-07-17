namespace UnitTestProject1

open System
open FSharpGPU
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
        let cudaResult1 = cudaArray |> DeviceArray.map <@ fun x -> x + 1.0 @> |> Array.ofCudaArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // test 2
        let array = Array.init (100) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cpuResult1 = array |> Array.map (fun x -> x - 19.0)
        let cudaResult1 = cudaArray |> DeviceArray.map <@ fun x -> x - 19.0 @> |> Array.ofCudaArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // test 3
        let array = Array.init (100) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cpuResult1 = array |> Array.map (fun x -> sin x - 19.0 * x )
        let cudaResult1 = cudaArray |> DeviceArray.map <@ fun x -> sin x - 19.0 * x @> |> Array.ofCudaArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // test 4
        let array = Array.init (100) (fun i -> rnd.NextDouble() * 10.0)
        let cudaArray = DeviceArray.ofArray array
        let cpuResult1 = array |> Array.map (fun x -> sqrt(x) / x /5.0 + 1.0/7.78)
        let cudaResult1 = cudaArray |> DeviceArray.map <@ fun x -> (sqrt(x) / x /5.0 + 1.0/7.78) @> |> Array.ofCudaArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        // test 5
        let array = Array.init (100) (fun i -> rnd.NextDouble() * 1.0e116)
        let cudaArray = DeviceArray.ofArray array
        let cpuResult1 = array |> Array.map (fun x -> x ** 5.731 *  x + 63784.0 - 938724.4)
        let cudaResult1 = cudaArray |> DeviceArray.map <@ fun x -> x ** 5.731 *  x + 63784.0 - 938724.4 @> |> Array.ofCudaArray
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
        let cudaResult1 = (cudaArray, cudaArray2) ||> DeviceArray.map2 <@ fun x y -> x + y @> |> Array.ofCudaArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        //test 2
        let array = Array.init (100) (fun i -> rnd.NextDouble())
        let array2 = Array.init (100) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult1 = (array, array2) ||> Array.map2 (fun x y -> x - y)
        let cudaResult1 = (cudaArray, cudaArray2) ||> DeviceArray.map2 <@ fun x y -> x - y @> |> Array.ofCudaArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        //test 3
        let array = Array.init (100) (fun i -> rnd.NextDouble())
        let array2 = Array.init (100) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult1 = (array, array2) ||> Array.map2 (fun x y -> sin x - cos y)
        let cudaResult1 = (cudaArray, cudaArray2) ||> DeviceArray.map2 <@ fun x y -> sin x - cos y @> |> Array.ofCudaArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))
        //test 4
        let array = Array.init (100) (fun i -> rnd.NextDouble())
        let array2 = Array.init (100) (fun i -> rnd.NextDouble())
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult1 = (array, array2) ||> Array.map2 (fun x y -> sqrt x -  y ** 11.27)
        let cudaResult1 = (cudaArray, cudaArray2) ||> DeviceArray.map2 <@ fun x y -> sqrt x - y ** 11.27 @> |> Array.ofCudaArray
        (cpuResult1, cudaResult1) ||> Array.iter2 (fun a1 a2 -> Assert.AreEqual(a1, a2, tolerance))