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
    |Stencil5 of Expr<'a->'a->'a->'a->'a->'b>
/// Behaviour when arrays are of differing lengths
type MappingLength =
    |Preserve
    |Shrink
/// Extra Device operations on arrays
[<RequireQualifiedAccess>]
type Array =
    /// Converts a device array into a standard host array
    static member ofDeviceArray (array : devicearray<devicefloat>) =
        let devArray = array.DeviceArray
        match devArray.ArrayType with
        |ComputeResult.ResComputeFloat v ->
            let hostArray = Array.zeroCreate<float> (devArray.Length)
            DeviceInterop.retrieveCUDADoubleArray(devArray.CudaPtr, devArray.Offset, hostArray, hostArray.Length) |> DeviceInterop.cudaCallWithExceptionCheck
            hostArray
        |_ -> failwith "Invalid type"

    /// Converts a device array into a standard host array
    static member ofDeviceArray (array : devicearray<devicebool>) =
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
    let rec private decomposeMap code (mapArgs : Map<Var,_>) =
        match code with
        // SPECIAL CASES
        | Double f ->
            ResComputeFloat f
        | Bool b ->
            ResComputeBool b
        |Var(var) ->
            mapArgs.[var]
        // SIMPLE OPERATORS
        |SpecificCall <@ (+) @> (_, _, [lhsExpr; rhsExpr]) -> // (+) Operator
            use lhs = decomposeMap lhsExpr mapArgs
            use rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapAdd lhs rhs
        |SpecificCall <@ (-) @> (_, _, [lhsExpr; rhsExpr]) -> // (-) Operator
            use lhs = decomposeMap lhsExpr mapArgs
            use rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapSubtract lhs rhs
        |SpecificCall <@ (*) @> (_, _, [lhsExpr; rhsExpr]) -> // (*) Operator
            use lhs = decomposeMap lhsExpr mapArgs
            use rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapMultiply lhs rhs
        |SpecificCall <@ (/) @> (_, _, [lhsExpr; rhsExpr]) -> // (/) Operator
            use lhs = decomposeMap lhsExpr mapArgs
            use rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapDivide lhs rhs
        |SpecificCall <@ ( ** ) @> (_, _, [lhsExpr; rhsExpr]) -> // (**) Operator
            use lhs = decomposeMap lhsExpr mapArgs
            use rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapPower lhs rhs
        |SpecificCall <@ sqrt @> (_, _, [expr]) -> // sqrt function
            use internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapSqrt internalExpr
        // TRIG FUNCTIONS
        |SpecificCall <@ cos @> (_, _, [expr]) -> // cos function
            use internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapCos internalExpr
        |SpecificCall <@ sin @> (_, _, [expr]) -> // sin function
            use internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapSin internalExpr
        |SpecificCall <@ tan @> (_, _, [expr]) -> // tan function
            use internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapTan internalExpr
        // HYPERBOLIC FUNCTIONS
        |SpecificCall <@ cosh @> (_, _, [expr]) -> // cosh function
            use internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapCosh internalExpr
        |SpecificCall <@ sinh @> (_, _, [expr]) -> // sinh function
            use internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapSinh internalExpr
        |SpecificCall <@ tanh @> (_, _, [expr]) -> // tanh function
            use internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapTanh internalExpr
        // INVERSE TRIG FUNCTIONS
        |SpecificCall <@ acos @> (_, _, [expr]) -> // acos function
            use internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapArcCos internalExpr
        |SpecificCall <@ asin @> (_, _, [expr]) -> // asin function
            use internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapArcSin internalExpr
        |SpecificCall <@ atan @> (_, _, [expr]) -> // tanh function
            use internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapArcTan internalExpr
        // LOG AND EXPONENTIAL FUNCTIONS
        |SpecificCall <@ log @> (_, _, [expr]) -> // log function
            use internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapLog internalExpr
        |SpecificCall <@ log10 @> (_, _, [expr]) -> // log10 function
            use internalExpr = decomposeMap expr mapArgs
            GeneralDeviceKernels.mapLog10 internalExpr
         // COMPARISON OPERATORS
        |SpecificCall <@ (.>.) : devicefloat -> float -> devicebool @> (_, _, [lhsExpr; rhsExpr])
        |SpecificCall <@ (.>.) : devicefloat -> devicefloat -> devicebool @> (_, _, [lhsExpr; rhsExpr])
        |SpecificCall <@ (.>.) : float -> devicefloat -> devicebool @> (_, _, [lhsExpr; rhsExpr])
            -> // (>) Operator
            use lhs = decomposeMap lhsExpr mapArgs
            use rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapGreaterThan lhs rhs
        |SpecificCall <@ (.>=.) : devicefloat -> float -> devicebool @> (_, _, [lhsExpr; rhsExpr]) 
        |SpecificCall <@ (.>=.) : devicefloat -> devicefloat -> devicebool @> (_, _, [lhsExpr; rhsExpr]) 
        |SpecificCall <@ (.>=.) : float -> devicefloat -> devicebool @> (_, _, [lhsExpr; rhsExpr]) 
            -> // (>=) Operator
            use lhs = decomposeMap lhsExpr mapArgs
            use rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapGreaterThanOrEqual lhs rhs
        |SpecificCall <@ (.<.) : devicefloat -> float -> devicebool @> (_, _, [lhsExpr; rhsExpr]) 
        |SpecificCall <@ (.<.) : devicefloat -> devicefloat -> devicebool @> (_, _, [lhsExpr; rhsExpr]) 
        |SpecificCall <@ (.<.) : float -> devicefloat -> devicebool @> (_, _, [lhsExpr; rhsExpr]) 
            -> // (<) Operator
            use lhs = decomposeMap lhsExpr mapArgs
            use rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapLessThan lhs rhs
        |SpecificCall <@ (.<=.) : devicefloat -> float -> devicebool @> (_, _, [lhsExpr; rhsExpr]) 
        |SpecificCall <@ (.<=.) : devicefloat -> devicefloat -> devicebool @> (_, _, [lhsExpr; rhsExpr]) 
        |SpecificCall <@ (.<=.) : float -> devicefloat -> devicebool @> (_, _, [lhsExpr; rhsExpr]) 
            -> // (<=) Operator
            use lhs = decomposeMap lhsExpr mapArgs
            use rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapLessThanOrEqual lhs rhs
        // EQUALITY OPERATORS
        |SpecificCall <@ (.=.) @> (_, _, [lhsExpr; rhsExpr]) -> // (=) Operator
            use lhs = decomposeMap lhsExpr mapArgs
            use rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapEquality lhs rhs
        |SpecificCall <@ (.<>.) @> (_, _, [lhsExpr; rhsExpr]) -> // (<>) Operator
            use lhs = decomposeMap lhsExpr mapArgs
            use rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapInequality lhs rhs
        |SpecificCall <@ (.&&.) @> (_, _, [lhsExpr; rhsExpr]) -> // (<>) Operator
            use lhs = decomposeMap lhsExpr mapArgs
            use rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapConditionalAnd lhs rhs
        |SpecificCall <@ (.||.) @> (_, _, [lhsExpr; rhsExpr]) -> // (<>) Operator
            use lhs = decomposeMap lhsExpr mapArgs
            use rhs = decomposeMap rhsExpr mapArgs
            GeneralDeviceKernels.mapConditionalOr lhs rhs
        // OTHER
        |_ -> failwith "Operation Not Supported."
    
    /// Create an offset array from a supplied array and the specified offset
    let private createArrayOffset offS newLength (array : ComputeArray) =
        match newLength with
        |None ->
            ComputeArray(array.ArrayType, array.CudaPtr, array.Length, OffsetSubarray(offS), AutoGenerated)
        |Some n ->
            ComputeArray(array.ArrayType, array.CudaPtr, n, OffsetSubarray(offS), AutoGenerated)

    //Higher order function for handling all mappings of N arguments
    let mapN code arrayList : ComputeArray =
        let rec mapAnyN code ( mapping : Map<_,_> ) arrayList =
            match code with
            |Lambda(var1, body) -> 
                match arrayList with
                |(currentArray :: remainingArrays) -> mapAnyN body (mapping.Add(var1, ResComputeArray(currentArray))) remainingArrays
                |_ -> raise <| System.InvalidOperationException("Mismatch between the number of device lambda arguments and the number of device arrays")
            |_ ->
                decomposeMap code mapping
        let result = mapAnyN code Map.empty arrayList
        match result with
            |ResComputeArray devArray -> devArray
            |_ -> failwith "Return type was not a device array"

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
            |NeighbourMapping.Stencil5 code -> // neighbour mapping of X_i, X_(i-2), X_(i-1), X_(i+1) and X_(i+2)
                let array2 = createArrayOffset -2 None array1
                let array3 = createArrayOffset -1 None array1
                let array4 = createArrayOffset 1 None array1
                let array5 = createArrayOffset 2 None array1
                let result = mapN code [array1; array2; array3; array4; array5]
                createArrayOrOffsetFromSpec mapLengthSpec 
                    (createArrayOffset 0 (Some <| array1.Length) result) // length preserving case
                    (createArrayOffset 2 (Some <| array1.Length-4) result) // shrinking case : 4 elements shorter with 2 positive offset
        devicearray<'b>(result) // convert typeless result to typed device array

    let filter (code : Expr<'a->devicebool>) (array : devicearray<'a>) =
        let result = mapN code [array.DeviceArray]
        let mutable length = 0
        let mutable cudaPtr = System.IntPtr.Zero
        DeviceInterop.createUninitialisedArray(array.DeviceArray.Length, DeviceArrayInfo.length array.DeviceArray.ArrayType, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
        DeviceFloatKernels.filter(array.DeviceArray.CudaPtr, result.CudaPtr, array.DeviceArray.Length, cudaPtr, &length) |> DeviceInterop.cudaCallWithExceptionCheck
        let arrayRes = ComputeArray(array.DeviceArray.ArrayType, cudaPtr, length, FullArray, UserGenerated)
        devicearray<'a>(arrayRes)

    /// Reduction functions
    module TypedReductions =
        let assocReduceFloat (code : Expr<devicefloat -> devicefloat -> devicefloat>) (array : devicearray<devicefloat>) =
            let rec assocReduceFloatIntrnl (code : Expr<devicefloat -> devicefloat -> devicefloat>) (array : ComputeArray) =
                match (array.Length) with
                |0 -> 
                    raise <| System.ArgumentException("array cannot be empty", "array")
                |1 ->
                    array |> devicearray<devicefloat> |> Array.ofDeviceArray |> Array.head
                |_ ->
                    let array2 = createArrayOffset 1 None array
                    let result = (mapN code [array; array2]) // evaluate reduction on X_i and X_(i+1)
                    let mutable cudaPtr = System.IntPtr(0)
                    DeviceInterop.createUninitialisedCUDADoubleArray((result.Length+1)/2, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
                    DeviceFloatKernels.reduceToHalf(result.CudaPtr, 0, result.Length, cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck // reduce the resulting array to half size
                    let newArr = ComputeArray(ComputeResult.ResComputeFloat(0.0), cudaPtr, (result.Length+1)/2, FullArray, AutoGenerated)
                    assocReduceFloatIntrnl code newArr // repeat until array of size 1
            assocReduceFloatIntrnl code array.DeviceArray

        let sumTotal (code : Expr<devicefloat -> devicefloat>) (array : devicearray<devicefloat>) =
            let result = (mapN code [array.DeviceArray])
            let mutable cudaPtr = System.IntPtr(0)
            DeviceInterop.createUninitialisedCUDADoubleArray(1, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
            DeviceFloatKernels.sumTotal(result.CudaPtr, 0, result.Length, cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck // reduce the resulting array to half size
            let resultArr = ComputeArray(ComputeResult.ResComputeFloat(0.0), cudaPtr, 1, FullArray, AutoGenerated)
            resultArr |> devicearray<devicefloat> |> Array.ofDeviceArray |> Array.head
            

/// Stencil templates
type Stencils =
    /// A neighbour mapping stencil of the form X_(i-1) and X_(i+1)
    static member Stencil2 ([<ReflectedDefinition>] code : Expr<'a->'a->'b>) =
        NeighbourMapping.Stencil2 code
    // A neighbour mapping stencil of the form X_i, X_(i-1) and X_(i+1)
    static member Stencil3 ([<ReflectedDefinition>] code : Expr<'a->'a->'a->'b>) =
        NeighbourMapping.Stencil3 code
    // A neighbour mapping stencil of the form X_i, X_(i-2), X_(i-1), X_(i+1) and X_(i+2)
    static member Stencil5 ([<ReflectedDefinition>] code : Expr<'a->'a->'a->'a->'a->'b>) =
        NeighbourMapping.Stencil5 code
    static member ImLeft ([<ReflectedDefinition>] code : Expr<'a->'a->'b>) =
        NeighbourMapping.ImmediateLeft code
    static member ImRight ([<ReflectedDefinition>] code : Expr<'a->'a->'b>) =
        NeighbourMapping.ImmediateRight code
    
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
        new devicearray<devicefloat>(ComputeArray(ComputeResult.ResComputeFloat(0.0), cudaPtr, Array.length array, FullArray, UserGenerated))
    /// Converts a standard host array into a device array
    static member ofArray (array : bool[]) =
        let mutable cudaPtr = System.IntPtr(0)
        DeviceInterop.initialiseCUDABoolArray(array, Array.length array, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
        new devicearray<devicefloat>(ComputeArray(ComputeResult.ResComputeBool(false), cudaPtr, Array.length array, FullArray, UserGenerated))
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
    // FILTERS
    // ----

    /// Returns a new array containing only the elements of the array for which the given predicate returns true.
    static member filter ([<ReflectedDefinition>] expr) =
        DeviceArrayOps.filter expr

    //
    // REDUCTIONS
    // ----------

    /// applies an associative reduction to the device array (the supplied function must be associative or this function will produce unexpected results)
    static member associativeReduce ([<ReflectedDefinition()>] code : Expr<devicefloat -> devicefloat -> devicefloat>) =
        DeviceArrayOps.TypedReductions.assocReduceFloat code


    static member sumBy ([<ReflectedDefinition()>] code : Expr<devicefloat  -> devicefloat>) =
        DeviceArrayOps.TypedReductions.sumTotal code



