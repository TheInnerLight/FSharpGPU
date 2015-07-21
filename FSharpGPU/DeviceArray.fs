﻿(*This file is part of FSharpGPU.

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

namespace NovelFS.FSharpGPU

open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.Patterns
open Microsoft.FSharp.Quotations.DerivedPatterns
open FSharp.Quotations.Evaluator
open ComputeOperators

/// Arguments for mappings of different lengths
type private MapArgs<'a> =
    |Map1Args of Var * ComputeResult
    |Map2Args of Var * ComputeResult * Var * ComputeResult
    |Map3Args of Var * ComputeResult* Var * ComputeResult * Var * ComputeResult
/// Cases for neighbour mapping
[<RequireQualifiedAccess>]
type NeighbourMapping<'a,'b> =
    |ImmediateLeft of code : Expr<'a->'a->'b>
    |ImmediateRight of Expr<'a->'a->'b>
    |Stencil2 of Expr<'a->'a->'b>
    |Stencil3 of Expr<'a->'a->'a->'b>
/// Behaviour when arrays are of differing lengths
type MappingLength =
    |Preserve
    |Shrink
/// Extra Device operations on arrays
[<RequireQualifiedAccess>]
type Array =
    /// Converts a device array into a standard host array
    static member ofCudaArray (array : devicearray<devicefloat>) =
        let devArray = array.DeviceArray
        match devArray.ArrayType with
        |ComputeResult.ResComputeFloat v ->
            let hostArray = Array.zeroCreate<float> (devArray.Length)
            DeviceInterop.retrieveCUDADoubleArray(devArray.CudaPtr, devArray.Offset, hostArray, hostArray.Length) |> DeviceInterop.cudaCallWithExceptionCheck
            hostArray
        |_ -> failwith "Invalid type"

    /// Converts a device array into a standard host array
    static member ofCudaArray (array : devicearray<devicebool>) =
        let devArray = array.DeviceArray
        match devArray.ArrayType with
        |ComputeResult.ResComputeBool v ->
            let hostArray = Array.zeroCreate<int> (devArray.Length)
            DeviceInterop.retrieveCUDABoolArray(devArray.CudaPtr, devArray.Offset, hostArray, hostArray.Length) |> DeviceInterop.cudaCallWithExceptionCheck
            hostArray |> Array.map (function
                                    |0 -> false
                                    |_ -> true)
        |_ -> failwith "Invalid type"

/// Basic operation implementation on Device Arrays
module private DeviceArrayOps =
    /// Returns the length of the device array
    let length (array : devicearray<'a>) =
        array.DeviceArray.Length
    /// recursively break apart the tree containing standard F# functions and recompose it using CUDA functions
    let rec private decomposeMap code (mapArgs : MapArgs<'a>) =
        match code with
        // SPECIAL CASES
        | Double f ->
            ResComputeFloat f
        | Bool b ->
            ResComputeBool b
        |Var(var) ->
            match mapArgs with
            |Map1Args(var1, array1) -> array1
            |Map2Args(var1, array1, var2, array2) ->
                if (var = var1) then array1 else array2
            |Map3Args(var1, array1, var2, array2, var3, array3) ->
                if (var = var1) then array1 elif (var = var2) then array2 else array3
        // SIMPLE OPERATORS
        |SpecificCall <@ (+) @> (_, _, [lhsExpr; rhsExpr]) -> // (+) Operator
            let lhs = decomposeMap lhsExpr mapArgs
            let rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapAdd lhs rhs
        |SpecificCall <@ (-) @> (_, _, [lhsExpr; rhsExpr]) -> // (-) Operator
            let lhs = decomposeMap lhsExpr mapArgs
            let rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapSubtract lhs rhs
        |SpecificCall <@ (*) @> (_, _, [lhsExpr; rhsExpr]) -> // (*) Operator
            let lhs = decomposeMap lhsExpr mapArgs
            let rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapMultiply lhs rhs
        |SpecificCall <@ (/) @> (_, _, [lhsExpr; rhsExpr]) -> // (/) Operator
            let lhs = decomposeMap lhsExpr mapArgs
            let rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapDivide lhs rhs
        |SpecificCall <@ ( ** ) @> (_, _, [lhsExpr; rhsExpr]) -> // (**) Operator
            let lhs = decomposeMap lhsExpr mapArgs
            let rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapPower lhs rhs
        |SpecificCall <@ sqrt @> (_, _, [expr]) -> // sqrt function
            let internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapSqrt internalExpr
        // TRIG FUNCTIONS
        |SpecificCall <@ cos @> (_, _, [expr]) -> // cos function
            let internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapCos internalExpr
        |SpecificCall <@ sin @> (_, _, [expr]) -> // sin function
            let internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapSin internalExpr
        |SpecificCall <@ tan @> (_, _, [expr]) -> // tan function
            let internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapTan internalExpr
        // HYPERBOLIC FUNCTIONS
        |SpecificCall <@ cosh @> (_, _, [expr]) -> // cosh function
            let internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapCosh internalExpr
        |SpecificCall <@ sinh @> (_, _, [expr]) -> // sinh function
            let internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapSinh internalExpr
        |SpecificCall <@ tanh @> (_, _, [expr]) -> // tanh function
            let internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapTanh internalExpr
        // INVERSE TRIG FUNCTIONS
        |SpecificCall <@ acos @> (_, _, [expr]) -> // acos function
            let internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapArcCos internalExpr
        |SpecificCall <@ asin @> (_, _, [expr]) -> // asin function
            let internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapArcSin internalExpr
        |SpecificCall <@ atan @> (_, _, [expr]) -> // tanh function
            let internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapArcTan internalExpr
        // LOG AND EXPONENTIAL FUNCTIONS
        |SpecificCall <@ log @> (_, _, [expr]) -> // log function
            let internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapLog internalExpr
        |SpecificCall <@ log10 @> (_, _, [expr]) -> // log10 function
            let internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapLog10 internalExpr
         // COMPARISON OPERATORS
        |SpecificCall <@ (.>.) @> (_, _, [lhsExpr; rhsExpr]) -> // (>) Operator
            let lhs = decomposeMap lhsExpr mapArgs
            let rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapGreaterThan lhs rhs
        |SpecificCall <@ (.>=.) @> (_, _, [lhsExpr; rhsExpr]) -> // (>=) Operator
            let lhs = decomposeMap lhsExpr mapArgs
            let rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapGreaterThanOrEqual lhs rhs
        |SpecificCall <@ (.<.) @> (_, _, [lhsExpr; rhsExpr]) -> // (<) Operator
            let lhs = decomposeMap lhsExpr mapArgs
            let rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapLessThan lhs rhs
        |SpecificCall <@ (.<=.) @> (_, _, [lhsExpr; rhsExpr]) -> // (<=) Operator
            let lhs = decomposeMap lhsExpr mapArgs
            let rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapLessThanOrEqual lhs rhs
        // EQUALITY OPERATORS
        |SpecificCall <@ (.=.) @> (_, _, [lhsExpr; rhsExpr]) -> // (=) Operator
            let lhs = decomposeMap lhsExpr mapArgs
            let rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapEquality lhs rhs
        |SpecificCall <@ (.<>.) @> (_, _, [lhsExpr; rhsExpr]) -> // (<>) Operator
            let lhs = decomposeMap lhsExpr mapArgs
            let rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapInequality lhs rhs
        // OTHER
        |_ -> failwith "Operation Not Supported."
    
    /// Create an offset array from a supplied array and the specified offset
    let private createArrayOffset offS newLength (array : ComputeArray) =
        match newLength with
        |None ->
            ComputeArray(array.ArrayType, array.CudaPtr, array.Length, OffsetSubarray(offS), AutoGenerated)
        |Some n ->
            ComputeArray(array.ArrayType, array.CudaPtr, n, OffsetSubarray(offS), AutoGenerated)

    /// Higher order function for handling all mappings of N arguments
    let private mapN code arrayList : ComputeArray =
        let result = 
            match code with
            |Lambda(var1, body) -> // 1 arg lambda
                let array1 = List.item 0 arrayList 
                match body with
                |Lambda (var2, body2) -> // 2 arg lambda
                    let array2 = List.item 1 arrayList
                    match body2 with
                    |Lambda (var3, body3) -> // 3 arg lambda
                        let array3 = List.item 2 arrayList
                        decomposeMap body3 (Map3Args(var1, ResComputeArray(array1), var2, ResComputeArray(array2), var3, ResComputeArray(array3)))
                    |_ ->
                        decomposeMap body2 (Map2Args(var1, ResComputeArray(array1), var2, ResComputeArray(array2)))
                |_ -> 
                    decomposeMap body (Map1Args(var1, ResComputeArray(array1)))
            |_ -> failwith "Not a valid map lambda"
        match result with
            |ResComputeArray devArray -> devArray
            |_ -> failwith "Return type was not a cuda array"

    /// Map involving 3 arrays
    let private map3 (code : Expr<'a->'a->'a->'b>) (array1 : devicearray<'a>) (array2 : devicearray<'a>) (array3 : devicearray<'a>) =
        mapN code [array1.DeviceArray; array2.DeviceArray; array3.DeviceArray]

    /// builds a new array whose elements are the results of applying the given function to each element of the array.
    let map (code : Expr<'a->'b>) (array : devicearray<'a>) =
        let result = mapN code [array.DeviceArray]
        devicearray<'b>(result)

    /// builds a new array whose elements are the results of applying the given function to each element of the array.
    let map2 (code : Expr<'a->'a->'b>) (array1 : devicearray<'a>) (array2 : devicearray<'a>) =
        let result = mapN code [array1.DeviceArray; array2.DeviceArray]
        devicearray<'b>(result)

    /// builds a new array whose elements are the results of applying the given function to each element of the array and a specified number of its neighbours
    let mapNeighbours (neighbourSpec : NeighbourMapping<'a,'b>) mapLengthSpec (inArray : devicearray<'a>) =
        /// creates a reference to an existing array with some kind of offset
        let createArrayOrOffsetFromSpec mapLengthSpec preserveCase shrinkCase = 
            match mapLengthSpec with 
            |Preserve ->
                preserveCase
            |Shrink ->
                shrinkCase
        let array1 = inArray.DeviceArray
        let result = // neighbour mapping logic: we create various copies of the array with offsets and length changes applied so that we can use map2, map3, etc. between them
            match neighbourSpec with
            |NeighbourMapping.ImmediateLeft code -> // neighbour mapping of X_i and X_(i-1)
                let array2 = createArrayOffset -1 None array1
                let result = mapN code [array1; array2]
                createArrayOrOffsetFromSpec mapLengthSpec 
                    (createArrayOffset 0 (Some <| array1.Length) result) // length preserving case
                    (createArrayOffset 1 (Some <| array1.Length-1) result) // shrinking case : 1 element shorter with 1 positive offset
            |NeighbourMapping.ImmediateRight code -> // neighbour mapping of X_i and X_(i+1)
                let array2 = createArrayOffset 1 None array1
                let result = mapN code [array1; array2]
                createArrayOrOffsetFromSpec mapLengthSpec 
                    (createArrayOffset 0 (Some <| array1.Length) result) // length preserving case
                    (createArrayOffset 0 (Some <| array1.Length-1) result) // shrinking case : 1 element shorter with 0 offset
            |NeighbourMapping.Stencil2 code -> // neighbour mapping of X_(i-1) and X_(i+1)
                let array2 = createArrayOffset -1 None array1
                let array3 = createArrayOffset 1 None array1
                let result = mapN code [array2; array3]
                createArrayOrOffsetFromSpec mapLengthSpec 
                    (createArrayOffset 0 (Some <| array1.Length) result) // length preserving case
                    (createArrayOffset 1 (Some <| array1.Length-2) result) // shrinking case : 2 elements shorter with 1 positive offset
            |NeighbourMapping.Stencil3 code -> // neighbour mapping of X_i, X_(i-1) and X_(i+1)
                let array2 = createArrayOffset -1 None array1
                let array3 = createArrayOffset 1 None array1
                let result = mapN code [array1; array2; array3]
                createArrayOrOffsetFromSpec mapLengthSpec 
                    (createArrayOffset 0 (Some <| array1.Length) result) // length preserving case
                    (createArrayOffset 1 (Some <| array1.Length-2) result) // shrinking case : 2 elements shorter with 1 positive offset
        devicearray<'b>(result) // convert typeless result to typed device array
    /// Reduction functions
    module TypedReductions =
        let assocReduceFloat (code : Expr<devicefloat -> devicefloat -> devicefloat>) (array : devicearray<devicefloat>) =
            let rec assocReduceFloatIntrnl (code : Expr<devicefloat -> devicefloat -> devicefloat>) (array : ComputeArray) =
                match (array.Length) with
                |0 -> 
                    raise <| System.ArgumentException("array cannot be empty", "array")
                |1 ->
                    array |> devicearray<devicefloat> |> Array.ofCudaArray |> Array.head
                |_ ->
                    let array2 = createArrayOffset 1 None array
                    let result = (mapN code [array; array2]) // evaluate reduction on X_i and X_(i+1)
                    let mutable cudaPtr = System.IntPtr(0)
                    DeviceInterop.createUninitialisedCUDADoubleArray((result.Length+1)/2, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
                    CudaFloatKernels.reduceToHalf(result.CudaPtr, 0, result.Length, cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck // reduce the resulting array to half size
                    let test = ComputeArray(ComputeResult.ResComputeFloat(0.0), cudaPtr, (result.Length+1)/2, FullArray, AutoGenerated)
                    assocReduceFloatIntrnl code test // repeat until array of size 1
            assocReduceFloatIntrnl code array.DeviceArray
/// Stencil templates
type Stencils =
    /// A neighbour mapping stencil of the form X_(i-1) and X_(i+1)
    static member Stencil2 ([<ReflectedDefinition>] code : Expr<'a->'a->'b>) =
        NeighbourMapping.Stencil2 code
    // A neighbour mapping stencil of the form X_i, X_(i-1) and X_(i+1)
    static member Stencil3 ([<ReflectedDefinition>] code : Expr<'a->'a->'a->'b>) =
        NeighbourMapping.Stencil3 code
    
//
// EXPOSED FUNCTIONS
// -----------------
/// Basic operations on Device Arrays
[<RequireQualifiedAccess>]
type DeviceArray =
    // UTILITY
    // -------

    /// Converts a standard host array into a device array
    static member ofArray (array : float[]) =
        let mutable cudaPtr = System.IntPtr(0)
        DeviceInterop.initialiseCUDADoubleArray(array, Array.length array, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
        devicearray<devicefloat>(ComputeArray(ComputeResult.ResComputeFloat(0.0), cudaPtr, Array.length array, FullArray, UserGenerated))
    /// Converts a standard host array into a device array
    static member ofArray (array : bool[]) =
        let mutable cudaPtr = System.IntPtr(0)
        DeviceInterop.initialiseCUDABoolArray(array, Array.length array, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
        devicearray<devicefloat>(ComputeArray(ComputeResult.ResComputeBool(false), cudaPtr, Array.length array, FullArray, UserGenerated))
    /// Returns the length of the device array
    static member length array =
        DeviceArrayOps.length array
    //
    // MAPS
    // ----

    /// Builds a new array whose elements are the results of applying the given function to each element of the array.
    static member map ([<ReflectedDefinition>] expr) =
        DeviceArrayOps.map expr
    /// Builds a new array whose elements are the results of applying the given function to each element of the array.
    static member map2 ([<ReflectedDefinition>] expr) =
        DeviceArrayOps.map2 expr
    /// Builds a new array whose elements are the results of applying the given function to each element of the array and a specified number of its neighbours
    static member mapNeighbours neighbourSpec mapLengthSpec array =
        DeviceArrayOps.mapNeighbours neighbourSpec mapLengthSpec array
    //
    // REDUCTIONS
    // ----------

    /// applies an associative reduction to the device array (the supplied function must be associative or this function will produce unexpected results)
    static member associativeReduce ([<ReflectedDefinition()>] code : Expr<devicefloat -> devicefloat -> devicefloat>) =
        DeviceArrayOps.TypedReductions.assocReduceFloat code


