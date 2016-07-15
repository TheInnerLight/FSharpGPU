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

open NovelFS.FSharpGPU
open FsCheck
open FsCheck.Xunit

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
        |0.0 -> -System.Double.Epsilon
        |_ -> System.BitConverter.Int64BitsToDouble(lng + 1L)

type ``Float Operator Equality Tests`` = 

    /// Unit tests for equality operator
    [<Property>]
    static member ``Map2 with equality between same arrays are all true`` (nArray : NormalFloat array) = 
        let array = nArray |> Array.map (fun x -> x.Get)
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = (cudaArray, cudaArray) ||> DeviceArray.map2 ( fun x y -> x .=. y ) |> Array.ofDeviceArray
        Array.forall (id) cudaResult

    /// Unit tests for inequality operator
    [<Property>]
    static member ``Map2 with inequality and + 1 between same arrays are all true`` (nArray : NormalFloat array) = 
        let array = nArray |> Array.map (fun x -> x.Get)
        let cudaArray = DeviceArray.ofArray array
        let cudaResult = (cudaArray, cudaArray) ||> DeviceArray.map2 ( fun x y -> (x+1.0) .<>. y ) |> Array.ofDeviceArray
        Array.forall (id) cudaResult


type ``Float Operator Comparison Tests``= 

    /// Unit tests for greater than operator
    [<Property>]
    static member ``Float array should always be greater than same float array with each element mapped to next smallest float`` (nArray : NormalFloat array) = 
        let array = nArray |> Array.map (fun x -> x.Get)
        let array2 = array |> Array.map (fun x ->  FloatHelper.decrFloatBySmallest x )
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> x .>. y ) |> Array.ofDeviceArray
        Array.forall (id) cudaResult

    /// Unit tests for greater than or equal operator
    [<Property>]
    static member  ``Float array should always be greater than or equal to same float array with each element mapped to next smallest float`` (nArray : NormalFloat array) = 
        let array = nArray |> Array.map (fun x -> x.Get)
        let array2 = array |> Array.map (fun x ->  FloatHelper.decrFloatBySmallest x )
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> x .>=. y ) |> Array.ofDeviceArray
        Array.forall (id) cudaResult

    [<Property>]
    static member ``Float array should always be less than same float array with each element mapped to next largest float`` (nArray : NormalFloat array) = 
        let array = nArray |> Array.map (fun x -> x.Get)
        let array2 = array |> Array.map (fun x ->  FloatHelper.incrFloatBySmallest x )
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> x .<. y ) |> Array.ofDeviceArray
        Array.forall (id) cudaResult

    /// Unit tests for greater than or equal operator
    [<Property>]
    static member ``Float array should always be less than or equal to same float array with each element mapped to next largest float`` (nArray : NormalFloat array) = 
        let array = nArray |> Array.map (fun x -> x.Get)
        let array2 = array |> Array.map (fun x ->  FloatHelper.incrFloatBySmallest x )
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 ( fun x y -> x .<=. y ) |> Array.ofDeviceArray
        Array.forall (id) cudaResult

type `` Float Operator Arithmetic Tests`` = 

    /// Unit tests for addition operator
    [<Property>]
    static member ``Map + between two float arrays produces same result on GPU and host`` (tArray : (float *float) array) = 
        let array, array2 = Array.unzip tArray
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult = (array, array2) ||> Array.map2 ( fun x y -> x + y )
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 (fun x y  -> x + y) |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.forall2 (TestConstants.checkFloatInRange TestConstants.tolerance)

    /// Unit tests for subtraction operator
    [<Property>]
    static member ``Map - between two float arrays produces same result on GPU and host`` (tArray : (float *float) array) = 
        let array, array2 = Array.unzip tArray 
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult = (array, array2) ||> Array.map2 ( fun x y -> x - y )
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 (fun x y  -> x - y) |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.forall2 (TestConstants.checkFloatInRange TestConstants.tolerance)

    /// Unit tests for multiplication operator
    [<Property>]
    static member ``Map * between two float arrays produces same result on GPU and host`` (tArray : (float *float) array) = 
        let array, array2 = Array.unzip tArray
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult = (array, array2) ||> Array.map2 ( fun x y -> x * y )
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 (fun x y  -> x * y) |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.forall2 (TestConstants.checkFloatInRange TestConstants.tolerance)

    /// Unit tests for division operator
    [<Property>]
    static member ``Map / between two float arrays produces same result on GPU and host`` (tArray : (float *float) array) = 
        let array, array2 = Array.unzip tArray 
        let cudaArray = DeviceArray.ofArray array
        let cudaArray2 = DeviceArray.ofArray array2
        let cpuResult = (array, array2) ||> Array.map2 ( fun x y -> x / y )
        let cudaResult = (cudaArray, cudaArray2) ||> DeviceArray.map2 (fun x y  -> x / y) |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.forall2 (TestConstants.checkFloatInRange TestConstants.tolerance)

    /// Unit tests for identity function
    [<Property>]
    static member ``Map id between two float arrays produces same result on GPU and host`` (array : float array) = 
        let cudaArray = DeviceArray.ofArray array
        let cpuResult = array |> Array.map (id )
        let cudaResult = cudaArray |> DeviceArray.map (id) |> Array.ofDeviceArray
        (cpuResult, cudaResult) ||> Array.forall2 (TestConstants.checkFloatInRange TestConstants.tolerance)