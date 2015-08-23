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
open Microsoft.FSharp.Quotations.ExprShape
open FSharp.Quotations.Evaluator

/// Arguments for mappings of different lengths
type private MapArgs<'a> =
    |Map1Args of Var * ComputeResult
    |Map2Args of Var * ComputeResult * Var * ComputeResult
    |Map3Args of Var * ComputeResult* Var * ComputeResult * Var * ComputeResult
/// Arguments for folds of different lengths
type private FoldArgs<'a> =
    |Fold1Args of Var * Var * ComputeResult
/// Cases for neighbour mapping
[<RequireQualifiedAccess>]
type NeighbourMapping<'a,'b> =
    /// A neighbour mapping stencil of the form X_i, and X_(i-1)
    |ImmediateLeft of code : Expr<'a->'a->'b>
    /// A neighbour mapping stencil of the form X_i, and X_(i+1)
    |ImmediateRight of Expr<'a->'a->'b>
    /// A neighbour mapping stencil of the form X_(i-1) and X_(i+1)
    |Stencil2 of Expr<'a->'a->'b>
    /// A neighbour mapping stencil of the form X_i, X_(i-1) and X_(i+1)
    |Stencil3 of Expr<'a->'a->'a->'b>
    /// A neighbour mapping stencil of the form X_i, X_(i-2), X_(i-1), X_(i+1) and X_(i+2)
    |Stencil5 of Expr<'a->'a->'a->'a->'a->'b>
/// Union cases for defining the behaviour when mapping over multiple elements in a stencil
type MappingLength =
    /// Preserve the length of the original array, treating any non-applicable values as zero
    |Preserve
    /// Shrink the result array to the number of elements over which the computation is valid
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

type SepFold =
    |VariableExpr of Var option * Expr
    |ManyVarExpr of Map<Var, Expr>

type SepFoldExpr =
    |MapExpr of System.Guid * Expr 
    |FoldExpr of Expr  * Expr list


/// Basic operation implementation on Device Arrays
module private DeviceArrayOps =
    /// Returns the length of the device array
    let length (array : devicearray<'a>) =
        array.DeviceArray.Length

//    let rec seperateFoldVariables foldVar code (array : devicearray<'a>)  =
//        match code with
//        |Value (_, _) -> 
//            VariableExpr (None, code)
//        |Var(var) -> 
//            VariableExpr (Some var, code)
//        |ShapeCombination(shapeComboObject, exprList) ->
//            let shapeMap = 
//                (Map.empty, exprList) ||> List.fold (fun tMap cExpr ->
//                    let exprRes = seperateFoldVariables foldVar cExpr array
//                    match exprRes with
//                    |VariableExpr (v, vExpr) ->
//                        match v with
//                        |Some v -> tMap |> Map.add v vExpr
//                        |None -> tMap
//                    |ManyVarExpr vMap ->
//                        Map.fold (fun acc key value -> Map.add key value acc) vMap tMap)
//            let test = Seq.head shapeMap
//            match shapeMap.ContainsKey(foldVar) with
//            |true ->
//                match shapeMap.Count with
//                |0 -> VariableExpr (None, code)
//                |1 ->
//                    let variable = shapeMap |> Map.toSeq |> Seq.head |> fst 
//                    VariableExpr(Some variable, code)
//            |false -> ManyVarExpr(shapeMap)
//
//        | ShapeLambda (var, expr) -> seperateFoldVariables foldVar expr array
    
    let rec seperateFoldVariables foldVar code (array : devicearray<'a>)  =
        let genGuid() = System.Guid.NewGuid()
        match code with
        |Value (_, _) -> 
            MapExpr(genGuid(), code) // An isolated value can be extracted as part of a map
        |Var(var) -> 
            match var = foldVar with
            |true -> FoldExpr(code, []) // If expression contains fold variable
            |false -> MapExpr(genGuid(), code) // Operations on variables other than the fold variable can be turned into maps
        |ShapeCombination(shapeComboObject, exprList) ->
            let subResults = exprList |> List.map (fun subExpr -> seperateFoldVariables foldVar subExpr array)
            match (subResults |> List.forall (function |MapExpr _ -> true; |FoldExpr _ -> false) ) with
            |true -> MapExpr (genGuid(), code)
            |false ->
                let exprAcc =
                    ([], subResults) ||> List.fold (fun acc subRes ->
                            match subRes with
                            |MapExpr (guid, mapCode) -> 
                                (Var(sprintf "`%A`" guid, mapCode.Type) |> Expr.Var, [mapCode] ) :: acc
                            |FoldExpr (foldCode, subCodes) ->
                                (foldCode, subCodes) :: acc)

                let exprList = exprAcc |> List.collect snd
                let combinedExpr = RebuildShapeCombination(shapeComboObject, exprAcc|> List.rev |> List.map fst)
                FoldExpr(combinedExpr, exprList)
                
        | ShapeLambda (var, expr) -> seperateFoldVariables foldVar expr array


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
        // IDENTITY
        |SpecificCall <@ id @> (_, _, [expr]) ->
            decomposeMap expr mapArgs
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

        let average (code : Expr<devicefloat -> devicefloat>) (array : devicearray<devicefloat>) =
            (sumTotal code array) / (float array.DeviceArray.Length)
            

/// A set of stencil templates for defining maps over several nearby array elements
type Stencils =
    /// A neighbour mapping stencil of the form X_(i-1) and X_(i+1)
    static member Stencil2 ([<ReflectedDefinition>] code : Expr<'a->'a->'b>) =
        NeighbourMapping.Stencil2 code
    /// A neighbour mapping stencil of the form X_i, X_(i-1) and X_(i+1)
    static member Stencil3 ([<ReflectedDefinition>] code : Expr<'a->'a->'a->'b>) =
        NeighbourMapping.Stencil3 code
    /// A neighbour mapping stencil of the form X_i, X_(i-2), X_(i-1), X_(i+1) and X_(i+2)
    static member Stencil5 ([<ReflectedDefinition>] code : Expr<'a->'a->'a->'a->'a->'b>) =
        NeighbourMapping.Stencil5 code
    /// A neighbour mapping stencil of the form X_i, and X_(i-1)
    static member ImLeft ([<ReflectedDefinition>] code : Expr<'a->'a->'b>) =
        NeighbourMapping.ImmediateLeft code
    /// A neighbour mapping stencil of the form X_i, and X_(i+1)
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

    /// Converts a standard host array of floats into a device array of devicefloats
    static member ofArray (array : float[]) =
        let mutable cudaPtr = System.IntPtr(0)
        DeviceInterop.initialiseCUDADoubleArray(array, Array.length array, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
        new devicearray<devicefloat>(ComputeArray(ComputeResult.ResComputeFloat(0.0), cudaPtr, Array.length array, FullArray, UserGenerated))
    /// Converts a standard host array of bools into a device array of devicebools
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
    //static member associativeReduce ([<ReflectedDefinition()>] code : Expr<devicefloat -> devicefloat -> devicefloat>) =
    //    DeviceArrayOps.TypedReductions.assocReduceFloat code

    /// Returns the sum of each element of the device array.
    static member sum() =
        DeviceArrayOps.TypedReductions.sumTotal <@ id : devicefloat -> devicefloat @>

    /// Returns the sum of the results generated by applying the function to each element of the device array.
    static member sumBy ([<ReflectedDefinition()>] code : Expr<devicefloat  -> devicefloat>) =
        DeviceArrayOps.TypedReductions.sumTotal code

    /// Returns the average of the elements generated by applying a function to each element of an array.
    static member averageBy ([<ReflectedDefinition()>] code : Expr<devicefloat  -> devicefloat>) =
        DeviceArrayOps.TypedReductions.average code

    /// Returns the average of the elements in the devicearray.
    static member average() =
        DeviceArrayOps.TypedReductions.average <@ id : devicefloat -> devicefloat  @>

    static member foldTest ([<ReflectedDefinition()>] code : Expr<float -> devicefloat -> devicefloat>) =
        match code with
        |ShapeLambda (var, expr) ->
            DeviceArrayOps.seperateFoldVariables var expr
        |_ -> failwith "Error"


