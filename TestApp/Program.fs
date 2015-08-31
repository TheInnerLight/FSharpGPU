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

open NovelFS.FSharpGPU

type TimerBuilder() =
    let watch = System.Diagnostics.Stopwatch()
    do watch.Start()
    let mutable lastMillis = 0L

    member this.Bind(v, f) = 
        let elapsedMillis = watch.ElapsedMilliseconds
        printfn "Time Taken : %d (ms)" (elapsedMillis - lastMillis)
        lastMillis <- elapsedMillis
        f v
    member this.Return(a) =
        a
    member this.Using(v, f) = 
        f v
    member this.ReturnFrom(a) =
        let elapsedMillis = watch.ElapsedMilliseconds
        printfn "Time Taken : %d (ms)" (elapsedMillis - lastMillis)
        lastMillis <- elapsedMillis
        a

[<EntryPoint>]
let main argv = 
    //let array = Array.init (16) (fun i -> (float i) + 2.0 )
    let array = Array.init (7500000) (fun i -> float i + 5.0)
    let array2 = Array.init (7500000) (fun i -> float (i*2))
    let array3 = Array.init 1500000 (fun i-> float i)
    let array4 = Array.init 1500000 (fun i-> float i)

    let timer = TimerBuilder();
    timer{
        do! ignore()
        let! cudaArray = DeviceArray.ofArray array
        let! cudaArray2 = DeviceArray.ofArray array2
        let! cudaArray3 = DeviceArray.ofArray array3
        let! cudaArray4 = DeviceArray.ofArray array4
        do! cudaArray |> DeviceArray.map id |> ignore

        let! testCPU2 = array |> Array.reduce (fun acc x -> acc + (sin x))

        printfn ""
        printfn "CUDA"
        printfn ""
       
        //let! resultm1 = cudaArray |> DeviceArray.map (fun x -> x + 2.0) |> Array.ofDeviceArray
        printfn "7.5 million element reduction"
        let! testReduce = cudaArray |> DeviceArray.associativeReduce (fun acc value -> acc + (6.0 * value))
        printfn "7.5 million element reduction"
        let! testReduce2 = cudaArray |> DeviceArray.associativeReduce (fun acc x -> acc + (sin x))
        printfn "7.5 million element (x2) map2 function"
        use! result = (cudaArray,cudaArray2) ||> DeviceArray.map2 (fun x y -> x ** y * sqrt y + 5.0 * sqrt y) |> Array.ofDeviceArray
        printfn "7.5 million element map function"
        use! result2 = cudaArray |> DeviceArray.map (fun x -> (sqrt(x) / x /5.0 + 1.0/7.78)) |> Array.ofDeviceArray
        printfn "7.5 million element filter"
        use! result2b = cudaArray |> DeviceArray.filter (fun x -> x .<. 100.0 ) |> Array.ofDeviceArray
        //let! result2 = cudaArray |> CudaArray.map (fun x -> x > 5.0 ) |> Array.ofCudaArray
        printfn "7.5 million element (x2) map2 function"
        use! result3 = (cudaArray,cudaArray2) ||> DeviceArray.map2 (fun x y ->  x * sqrt y .>. 123.5)
        printfn "1.5 million element summation function"
        use! result4 = cudaArray4 |> DeviceArray.sumBy (fun x -> x + 1.0)
        //let! result3a = cudaArray |> DeviceArray.map (fun x -> x) |> Array.ofDeviceArray
        //let! result3a = cudaArray |> DeviceArray.mapNeighbours (Stencils.Stencil3 (fun x l r -> x + 0.2 * l + 0.2 * r)) Preserve |> Array.ofDeviceArray
        //let! result3a = cudaArray |> DeviceArray.mapNeighbours (Stencils.Stencil3 (fun x l r -> x + 0.2*l + 0.2*r)) Preserve |> Array.ofDeviceArray
        //let! result4 = cudaArray3 |> DeviceArray.associativeReduce (fun x y ->  x + y )
        printfn ""
        printfn "CPU"
        printfn ""
        //
        printfn "7.5 million element reduction"
        let! testReduceCPU = array |> Array.reduce (fun acc value -> acc + (6.0 * value))
        printfn "7.5 million element reduction"
        let! testReduceCPU2 = array |> Array.reduce (fun acc x -> acc + (sin x))
        printfn "7.5 million element (x2) map2 function"
        let! resultCPU = (array, array2) ||> Array.map2 (fun x y -> x ** y * sqrt y + 5.0 * sqrt y ) 
        printfn "7.5 million element map function"
        let! result2CPU = array |> Array.map (fun x -> sqrt(x) / x /5.0 + 1.0/7.78)
        printfn "7.5 million element filter"
        let! result2bCPU = array |> Array.filter (fun x -> x < 100.0 )
        printfn "7.5 million element (x2) map2 function"
        let! result3CPU = (array, array2) ||> Array.map2 (fun x y -> x * sqrt y > 123.5  ) 
        printfn "1.5 million element summation function"
        let! result4CPU = array4 |> Array.sumBy (fun x -> x + 1.0)
        //let! result4CPU = array3 |> Array.reduce (fun x y -> x + y )
        printfn "..."
        //printfn "%A" result
        //printfn "%A" resultCPU
        //printfn "%A" result2
        //printfn "%A" result2CPU
        //printfn "%A" result4CPU
       // printfn "%A" result4
        //printfn "%A" result4
        printfn "%A" result2b
        printfn "%A" result2bCPU
        printfn "%A" array
        printfn "%A" result4
        printfn "%A" result4CPU
        return ()
        }
    0 // return an integer exit code
