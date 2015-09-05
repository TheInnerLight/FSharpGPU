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

type private SepFoldExpr =
    |MapExpr of System.Guid * Expr 
    |ReduceExpr of Expr  * Expr list * Var list

/// Basic operation implementation on Device Arrays
module private DeviceArrayOps =
    /// Returns the length of the device array
    let length (array : devicearray<'a>) =
        array.DeviceArray.Length

    let private fillFloat length value =
        let mutable cudaPtr = System.IntPtr(0)
        DeviceInterop.createUninitialisedCUDADoubleArray(length, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
        DeviceFloatKernels.setAllElementsToConstant(cudaPtr, 0, length, 0.0) |> DeviceInterop.cudaCallWithExceptionCheck
        ComputeArray(ComputeResult.ResComputeFloat(0.0), cudaPtr, length, FullArray, UserGenerated)

    let zeroCreate<'a when 'a :> IGPUType> (length : int) =
        match typeof<'a> with
        |x when x = typeof<devicefloat> ->
            new devicearray<'a>(fillFloat length 0.0)
        |_ ->
            failwith "No other types currently supported"

    let rec reApplyLambdas originalExpr varList newExpr =
        match originalExpr with
        |ShapeLambda (var, expr) ->
            reApplyLambdas expr (var :: varList) newExpr
        |_ ->
            let acc = Expr.Lambda(varList |> List.head, newExpr)
            (acc, varList |> List.tail) ||> List.fold (fun acc v -> Expr.Lambda(v, acc))
            
    
    let rec seperateReductionVariable foldVar code (array : devicearray<'a>)  =
        let genGuid() = System.Guid.NewGuid()
        match code with
        |Value (_, _) -> 
            MapExpr(genGuid(), code) // An isolated value can be extracted as part of a map
        |Var(var) -> 
            match var = foldVar with
            |true -> ReduceExpr(code, [], []) // If expression contains the reduction variable
            |false -> MapExpr(genGuid(), code) // Operations on variables other than the reduction variable can be turned into maps
        |ShapeCombination(shapeComboObject, exprList) ->
            let subResults = exprList |> List.map (fun subExpr -> seperateReductionVariable foldVar subExpr array)
            match (subResults |> List.forall (function |MapExpr _ -> true; |ReduceExpr _ -> false) ) with
            |true -> MapExpr (genGuid(), code)
            |false ->
                let exprAcc =
                    ([], subResults) ||> List.fold (fun acc subRes ->
                            match subRes with
                            |MapExpr (guid, mapCode) -> 
                                let var = Var(sprintf "`%A`" guid, mapCode.Type)
                                (var|> Expr.Var, [mapCode], [var]) :: acc
                            |ReduceExpr (foldCode, subCodes, varList) ->
                                (foldCode, subCodes, varList) :: acc)

                let exprList = exprAcc |> List.collect (fun (_, b, _) -> b)
                let guidList = exprAcc |> List.collect (fun (_, _, c) -> c)
                let combinedExpr = RebuildShapeCombination(shapeComboObject, exprAcc|> List.rev |> List.map (fun (a, _, _) -> a))
                ReduceExpr(combinedExpr, exprList, guidList)
                
        | ShapeLambda (var, expr) -> seperateReductionVariable foldVar expr array


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

    /// builds a new array whose elements are the results of applying the given function to each element of the array.
    let map (code : Expr<'a->'b>) (array : devicearray<'a>) =
        let result = mapN code [array.DeviceArray]
        devicearray<'b>(result)

    /// builds a new array whose elements are the results of applying the given function to each element of the array.
    let map2 (code : Expr<'a->'a->'b>) (array1 : devicearray<'a>) (array2 : devicearray<'a>) =
        let result = mapN code [array1.DeviceArray; array2.DeviceArray]
        devicearray<'b>(result)

    /// Map involving 3 arrays
    let private map3 (code : Expr<'a->'a->'a->'b>) (array1 : devicearray<'a>) (array2 : devicearray<'a>) (array3 : devicearray<'a>) =
        mapN code [array1.DeviceArray; array2.DeviceArray; array3.DeviceArray]

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

    /// filters the array using a stable filter
    let filter (code : Expr<'a->devicebool>) (array : devicearray<'a>) =
        let result = mapN code [array.DeviceArray]
        let mutable length = 0
        let mutable cudaPtr = System.IntPtr.Zero
        DeviceInterop.createUninitialisedArray(array.DeviceArray.Length, DeviceArrayInfo.length array.DeviceArray.ArrayType, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
        DeviceFloatKernels.filter(array.DeviceArray.CudaPtr, result.CudaPtr, array.DeviceArray.Length, cudaPtr, &length) |> DeviceInterop.cudaCallWithExceptionCheck
        let arrayRes = ComputeArray(array.DeviceArray.ArrayType, cudaPtr, length, FullArray, UserGenerated)
        devicearray<'a>(arrayRes)

    let evaluateMapsAndReconstructReduction (code : Expr<'a -> 'b -> 'a>) (array : devicearray<'b>) =
        match code with
        |ShapeLambda (var, expr) ->
            let foldResults = seperateReductionVariable var expr array
            match foldResults with
            |ReduceExpr (foldExpr, mapExrList, varList) ->
                let mapResults = mapExrList |> List.map (fun mapExpr -> 
                    let funWithLambda = reApplyLambdas expr [] mapExpr
                    mapN funWithLambda [array.DeviceArray])

                let acc = Expr.Lambda(varList |> List.head, foldExpr)
                let mapLambdas = (acc, varList |> List.tail) ||> List.fold (fun acc v -> Expr.Lambda(v, acc))
                (Expr.Lambda(var, mapLambdas), mapResults)
                //foldExpr
            |_ -> failwith "Error"
        |_ -> failwith "Error"

    /// Reduction functions
    module TypedReductions =
        /// The reduction function on floating point numbers
        let assocReduceFloat code (array : devicearray<devicefloat>) =
            /// Apply reduction to element 1 and element 2 and reduce the result to an array of half size
            let offsetMap (code : Expr<devicefloat -> devicefloat -> devicefloat>) (array1 : ComputeArray) (array2 : ComputeArray) =
                let result = (mapN code [array1; array2]) 
                let mutable cudaPtr = System.IntPtr(0)
                let len = if result.Length % 2 = 0 then result.Length / 2 else result.Length / 2 + 1
                DeviceInterop.createUninitialisedCUDADoubleArray(len, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
                DeviceFloatKernels.reduceToHalf(result.CudaPtr, 0, result.Length, cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck // reduce the resulting array to half size
                ComputeArray(ComputeResult.ResComputeFloat(0.0), cudaPtr, len, FullArray, AutoGenerated)
            /// Recursively apply the reduction to element X_i and X_i+1 and merge the results until only one element remains
            let rec assocReduceFloatIntrnl (code : Expr<devicefloat -> devicefloat -> devicefloat>) (array : ComputeArray) =
                match (array.Length) with
                |0 -> 
                    raise <| System.ArgumentException("array cannot be empty", "array")
                |1 ->
                    array //|> devicearray<devicefloat> |> Array.ofDeviceArray |> Array.head
                |_ ->
                    let array1 = createArrayOffset 0 (None) array
                    let array2 = createArrayOffset 1 (None) array
                    let newArr = offsetMap code array1 array2 // evaluate reduction on X_i and X_i+1
                    array.Dispose()
                    assocReduceFloatIntrnl code newArr // repeat until array of size 1
            // seperate off first element, apply maps to the remaining elements and merge the results
            let arrayMinusFirst = createArrayOffset 1 (None) array.DeviceArray |> devicearray<devicefloat>
            let foldExpr, mapResults = evaluateMapsAndReconstructReduction code arrayMinusFirst
            match List.length mapResults with
            |1 ->
                let foldExpr = Expr<devicefloat -> devicefloat -> devicefloat>.Cast foldExpr
                let devArray = List.head mapResults
                let devArray = createArrayOffset 0 (Some <| devArray.Length - 1) devArray
                let fstDevArray = createArrayOffset 0 (Some 1) array.DeviceArray
                let tailDevArray = assocReduceFloatIntrnl (foldExpr) devArray
                offsetMap foldExpr fstDevArray tailDevArray |> devicearray<devicefloat> |> Array.ofDeviceArray |> Array.head
            |_ ->
                raise <| System.InvalidOperationException("Reduction operation ended in an invalid state.")

        /// Function for summation of floating point numbers
        let sumTotal (code : Expr<devicefloat -> devicefloat>) (array : devicearray<devicefloat>) =
            let result = (mapN code [array.DeviceArray])
            let mutable cudaPtr = System.IntPtr(0)
            DeviceInterop.createUninitialisedCUDADoubleArray(1, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
            DeviceFloatKernels.sumTotal(result.CudaPtr, 0, result.Length, cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck // reduce the resulting array to half size
            let resultArr = ComputeArray(ComputeResult.ResComputeFloat(0.0), cudaPtr, 1, FullArray, AutoGenerated)
            resultArr |> devicearray<devicefloat> |> Array.ofDeviceArray |> Array.head

        /// Function for averaging of floating point numbers
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
    /// Creates a device array of all zero elements
    static member zeroCreate length =
        DeviceArrayOps.zeroCreate length


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

    /// Returns a new array containing only the elements of the array for which the given predicate returns true.  This operation performs a stable filter, i.e. does not change the order
    /// of the elements.
    static member filter ([<ReflectedDefinition>] expr) =
        DeviceArrayOps.filter expr

    //
    // REDUCTIONS
    // ----------

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

    /// Applies a function to each element of in the array and merging the results into an array of half size recursively until all elements in the array have been merged.
    /// Note: The function which merges the accumulator and the element MUST be associative or this function will produce unexpected results.
    static member associativeReduce ([<ReflectedDefinition()>] code : Expr<devicefloat -> devicefloat -> devicefloat>) =
        DeviceArrayOps.TypedReductions.assocReduceFloat code

