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

namespace NovelFS.FSharpGPU

open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.Patterns
open Microsoft.FSharp.Quotations.DerivedPatterns
open Microsoft.FSharp.Quotations.ExprShape

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
    static member inline ofDeviceArray array = DeviceHostTransfer.copyArrayToHost array

type private SepFoldExpr =
    |MapExpr of System.Guid * Expr 
    |ReduceExpr of Expr  * Expr list * Var list

/// Basic operation implementation on Device Arrays
module private DeviceArrayOps =
    /// Returns the length of the device array
    let length (array : devicearray<'a>) = ComputeResults.length (array.DeviceArrays)

    /// Create an array of default values of the specified length
    let defaultCreate<'a when 'a :> IGPUType> (length : int) =
        match typeof<'a> with
        |x when x = typeof<devicefloat> ->
            new devicearray<'a>(ComputeArrays.fillFloat length 0.0)
        |x when x = typeof<devicebool> ->
            new devicearray<'a>(ComputeArrays.fillBool length false)
        |_ ->
            failwith "No other types currently supported"

    /// Combines the two arrays into an array of tuples with two elements.
    let zip (array1 : devicearray<'a>) (array2 : devicearray<'b>) =
        match array1.DeviceArrays, array2.DeviceArrays with
        |ResComputeArray devArray1, ResComputeArray devArray2 -> new devicearray<'a*'b>([devArray1; devArray2])
        |ResEmpty, ResEmpty -> new devicearray<'a*'b>(ResEmpty)

    /// Combines the three arrays into an array of tuples with three elements.
    let zip3 (array1 : devicearray<'a>) (array2 : devicearray<'b>) (array3 : devicearray<'c>) =
        match array1.DeviceArrays, array2.DeviceArrays, array3.DeviceArrays with
        |ResComputeArray devArray1, ResComputeArray devArray2, ResComputeArray devArray3 -> 
            new devicearray<'a*'b*'c>([devArray1; devArray2; devArray3])

    /// Splits an array of pairs into two arrays.
    let unzip<'a,'b> (array : devicearray<'a*'b>) =
        let (dev1, dev2) = ComputeResult.assumePair (array.DeviceArrays)
        new devicearray<'a>(dev1), new devicearray<'b>(dev2)

    /// Splits an array of triples into three arrays.
    let unzip3<'a,'b,'c> (array : devicearray<'a*'b*'c>) =
        let (dev1, dev2, dev3) = ComputeResult.assumeTriple (array.DeviceArrays)
        new devicearray<'a>(dev1), new devicearray<'b>(dev2), new devicearray<'b>(dev3)

    /// Re-applys the lambdas from the start of a reduction expression to the map expressions
    let rec reApplyLambdas originalExpr varList newExpr =
        match originalExpr with
        |ShapeLambda (var, expr) ->
            reApplyLambdas expr (var :: varList) newExpr
        |_ ->
            let acc = Expr.Lambda(varList |> List.head, newExpr)
            (acc, varList |> List.tail) ||> List.fold (fun acc v -> Expr.Lambda(v, acc))
            
    /// Seperates the reduction variable from the constant variables in an expression
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
                    ([], subResults) 
                    ||> List.fold (fun acc subRes ->
                            match subRes with
                            |MapExpr (guid, mapCode) -> 
                                let var = Var(sprintf "`%A`" guid, mapCode.Type)
                                (Expr.Var var, [mapCode], [var]) :: acc
                            |ReduceExpr (foldCode, subCodes, varList) ->
                                (foldCode, subCodes, varList) :: acc)
                let exprList = exprAcc |> List.collect (fun (_, b, _) -> b)
                let guidList = exprAcc |> List.collect (fun (_, _, c) -> c)
                let combinedExpr = RebuildShapeCombination(shapeComboObject, exprAcc|> List.rev |> List.map (fun (a, _, _) -> a))
                ReduceExpr(combinedExpr, exprList, guidList)
        | ShapeLambda (var, expr) -> seperateReductionVariable foldVar expr array


    /// recursively break apart the tree containing standard F# functions and recompose it using CUDA functions
    let rec decomposeMap code (variableTable : Map<Var,_>) =
        match code with
        // SPECIAL CASES
        | Double f ->
            ResComputeFloat f
        | Bool b ->
            ResComputeBool b
        |Var(var) ->
            variableTable.[var]
        |Let(var, letBoundExpr, body) ->
            use result = decomposeMap letBoundExpr variableTable
            let newMapArgs = variableTable |> Map.add var (result)
            decomposeMap body newMapArgs
        |TupleGet (expr, i) ->
            match decomposeMap expr variableTable with
            |ResComputeTupleArray lst -> lst.[i]
            |_ -> raise <| System.InvalidOperationException()
        |NewTuple (exprList) ->
            exprList 
            |> List.map (fun cd -> decomposeMap cd variableTable)
            |> ResComputeTupleArray
        // IDENTITY
        |SpecificCall <@ id @> (_, _, [expr]) ->
            decomposeMap expr variableTable
        // TYPE COERSION
        |SpecificCall <@ devicefloat @> (_, _, [expr]) ->
            ResComputeFloat (ComputeResult.assumeFloat <| decomposeMap expr variableTable)
        |SpecificCall <@ devicebool @> (_, _, [expr]) ->
            ResComputeBool (ComputeResult.assumeBool <| decomposeMap expr variableTable)
        // SIMPLE OPERATORS
        |SpecificCall <@ (+) @> (_, _, [lhsExpr; rhsExpr]) -> // (+) Operator
            use lhs = decomposeMap lhsExpr variableTable
            use rhs = decomposeMap rhsExpr variableTable
            GeneralDeviceKernels.mapAdd lhs rhs
        |SpecificCall <@ (-) @> (_, _, [lhsExpr; rhsExpr]) -> // (-) Operator
            use lhs = decomposeMap lhsExpr variableTable
            use rhs = decomposeMap rhsExpr variableTable
            GeneralDeviceKernels.mapSubtract lhs rhs
        |SpecificCall <@ (*) @> (_, _, [lhsExpr; rhsExpr]) -> // (*) Operator
            use lhs = decomposeMap lhsExpr variableTable
            use rhs = decomposeMap rhsExpr variableTable
            GeneralDeviceKernels.mapMultiply lhs rhs
        |SpecificCall <@ (/) @> (_, _, [lhsExpr; rhsExpr]) -> // (/) Operator
            use lhs = decomposeMap lhsExpr variableTable
            use rhs = decomposeMap rhsExpr variableTable
            GeneralDeviceKernels.mapDivide lhs rhs
        |SpecificCall <@ ( ** ) @> (_, _, [lhsExpr; rhsExpr]) -> // (**) Operator
            use lhs = decomposeMap lhsExpr variableTable
            use rhs = decomposeMap rhsExpr variableTable
            GeneralDeviceKernels.mapPower lhs rhs
        |SpecificCall <@ sqrt @> (_, _, [expr]) -> // sqrt function
            use internalExpr = decomposeMap expr variableTable
            GeneralDeviceKernels.mapSqrt internalExpr
        // TRIG FUNCTIONS
        |SpecificCall <@ cos @> (_, _, [expr]) -> // cos function
            use internalExpr = decomposeMap expr variableTable
            GeneralDeviceKernels.mapCos internalExpr
        |SpecificCall <@ sin @> (_, _, [expr]) -> // sin function
            use internalExpr = decomposeMap expr variableTable
            GeneralDeviceKernels.mapSin internalExpr
        |SpecificCall <@ tan @> (_, _, [expr]) -> // tan function
            use internalExpr = decomposeMap expr variableTable
            GeneralDeviceKernels.mapTan internalExpr
        // HYPERBOLIC FUNCTIONS
        |SpecificCall <@ cosh @> (_, _, [expr]) -> // cosh function
            use internalExpr = decomposeMap expr variableTable
            GeneralDeviceKernels.mapCosh internalExpr
        |SpecificCall <@ sinh @> (_, _, [expr]) -> // sinh function
            use internalExpr = decomposeMap expr variableTable
            GeneralDeviceKernels.mapSinh internalExpr
        |SpecificCall <@ tanh @> (_, _, [expr]) -> // tanh function
            use internalExpr = decomposeMap expr variableTable
            GeneralDeviceKernels.mapTanh internalExpr
        // INVERSE TRIG FUNCTIONS
        |SpecificCall <@ acos @> (_, _, [expr]) -> // acos function
            use internalExpr = decomposeMap expr variableTable
            GeneralDeviceKernels.mapArcCos internalExpr
        |SpecificCall <@ asin @> (_, _, [expr]) -> // asin function
            use internalExpr = decomposeMap expr variableTable
            GeneralDeviceKernels.mapArcSin internalExpr
        |SpecificCall <@ atan @> (_, _, [expr]) -> // tanh function
            use internalExpr = decomposeMap expr variableTable
            GeneralDeviceKernels.mapArcTan internalExpr
        // LOG AND EXPONENTIAL FUNCTIONS
        |SpecificCall <@ log @> (_, _, [expr]) -> // log function
            use internalExpr = decomposeMap expr variableTable
            GeneralDeviceKernels.mapLog internalExpr
        |SpecificCall <@ log10 @> (_, _, [expr]) -> // log10 function
            use internalExpr = decomposeMap expr variableTable
            GeneralDeviceKernels.mapLog10 internalExpr
         // COMPARISON OPERATORS
        |SpecificCall <@ (.>.) : devicefloat -> float -> devicebool @> (_, _, [lhsExpr; rhsExpr])
        |SpecificCall <@ (.>.) : devicefloat -> devicefloat -> devicebool @> (_, _, [lhsExpr; rhsExpr])
        |SpecificCall <@ (.>.) : float -> devicefloat -> devicebool @> (_, _, [lhsExpr; rhsExpr])
            -> // (>) Operator
            use lhs = decomposeMap lhsExpr variableTable
            use rhs = decomposeMap rhsExpr variableTable
            GeneralDeviceKernels.mapGreaterThan lhs rhs
        |SpecificCall <@ (.>=.) : devicefloat -> float -> devicebool @> (_, _, [lhsExpr; rhsExpr]) 
        |SpecificCall <@ (.>=.) : devicefloat -> devicefloat -> devicebool @> (_, _, [lhsExpr; rhsExpr]) 
        |SpecificCall <@ (.>=.) : float -> devicefloat -> devicebool @> (_, _, [lhsExpr; rhsExpr]) 
            -> // (>=) Operator
            use lhs = decomposeMap lhsExpr variableTable
            use rhs = decomposeMap rhsExpr variableTable
            GeneralDeviceKernels.mapGreaterThanOrEqual lhs rhs
        |SpecificCall <@ (.<.) : devicefloat -> float -> devicebool @> (_, _, [lhsExpr; rhsExpr]) 
        |SpecificCall <@ (.<.) : devicefloat -> devicefloat -> devicebool @> (_, _, [lhsExpr; rhsExpr]) 
        |SpecificCall <@ (.<.) : float -> devicefloat -> devicebool @> (_, _, [lhsExpr; rhsExpr]) 
            -> // (<) Operator
            use lhs = decomposeMap lhsExpr variableTable
            use rhs = decomposeMap rhsExpr variableTable
            GeneralDeviceKernels.mapLessThan lhs rhs
        |SpecificCall <@ (.<=.) : devicefloat -> float -> devicebool @> (_, _, [lhsExpr; rhsExpr]) 
        |SpecificCall <@ (.<=.) : devicefloat -> devicefloat -> devicebool @> (_, _, [lhsExpr; rhsExpr]) 
        |SpecificCall <@ (.<=.) : float -> devicefloat -> devicebool @> (_, _, [lhsExpr; rhsExpr]) 
            -> // (<=) Operator
            use lhs = decomposeMap lhsExpr variableTable
            use rhs = decomposeMap rhsExpr variableTable
            GeneralDeviceKernels.mapLessThanOrEqual lhs rhs
        // EQUALITY OPERATORS
        |SpecificCall <@ (.=.) @> (_, _, [lhsExpr; rhsExpr]) -> // (=) Operator
            use lhs = decomposeMap lhsExpr variableTable
            use rhs = decomposeMap rhsExpr variableTable
            GeneralDeviceKernels.mapEquality lhs rhs
        |SpecificCall <@ (.<>.) @> (_, _, [lhsExpr; rhsExpr]) -> // (<>) Operator
            use lhs = decomposeMap lhsExpr variableTable
            use rhs = decomposeMap rhsExpr variableTable
            GeneralDeviceKernels.mapInequality lhs rhs
        |SpecificCall <@ (.&&.) @> (_, _, [lhsExpr; rhsExpr]) -> // (<>) Operator
            use lhs = decomposeMap lhsExpr variableTable
            use rhs = decomposeMap rhsExpr variableTable
            GeneralDeviceKernels.mapConditionalAnd lhs rhs
        |SpecificCall <@ (.||.) @> (_, _, [lhsExpr; rhsExpr]) -> // (<>) Operator
            use lhs = decomposeMap lhsExpr variableTable
            use rhs = decomposeMap rhsExpr variableTable
            GeneralDeviceKernels.mapConditionalOr lhs rhs
        // OTHER
        |_ -> failwith "Operation Not Supported."
    
    //Higher order function for handling all mappings of N arguments
    let mapN code arrayList =
        let rec mapAnyN code ( mapping : Map<_,_> ) arrayList =
            match code with
            |Lambda(var1, body) -> 
                match arrayList with
                |(currentArray :: remainingArrays) -> mapAnyN body (mapping.Add(var1, currentArray)) remainingArrays
                |_ -> raise <| System.InvalidOperationException("Mismatch between the number of device lambda arguments and the number of device arrays")
            |_ ->
                decomposeMap code mapping
        let length = ComputeResults.length <| List.head arrayList
        // if the result is a constant, expand it into an array of the appropriate type
        ComputeResults.expandValueToArray length (mapAnyN code Map.empty arrayList)

    /// builds a new array whose elements are the results of applying the given function to each element of the array.
    let map (code : Expr<'a->'b>) (array : devicearray<'a>) =
        match length array with
        |0 -> new devicearray<'b>(ResEmpty)
        |_ ->
            let result = mapN code [array.DeviceArrays]
            new devicearray<'b>(result)

    /// builds a new array whose elements are the results of applying the given function to each element of the array.
    let map2 (code : Expr<'a->'a->'b>) (array1 : devicearray<'a>) (array2 : devicearray<'a>) =
        match length array1, length array2 with
        |0,_ -> new devicearray<'b>(ResEmpty)
        |_,0 -> new devicearray<'b>(ResEmpty)
        |_ -> 
            let result = mapN code [array1.DeviceArrays; array2.DeviceArrays]
            new devicearray<'b>(result)


    /// Map involving 3 arrays
    let map3 (code : Expr<'a->'a->'a->'b>) (array1 : devicearray<'a>) (array2 : devicearray<'a>) (array3 : devicearray<'a>) =
        let result = mapN code [array1.DeviceArrays; array2.DeviceArrays; array3.DeviceArrays]
        new devicearray<'b>(result)

    /// builds a new array whose elements are the results of applying the given function to each element of the array and a specified number of its neighbours
    let mapNeighbours (neighbourSpec : NeighbourMapping<'a,'b>) mapLengthSpec (inArray : devicearray<'a>) =
        /// creates a reference to an existing array with some kind of offset
        let createArrayOrOffsetFromSpec mapLengthSpec preserveCase shrinkCase = 
            match mapLengthSpec with 
            |Preserve ->
                preserveCase
            |Shrink ->
                shrinkCase
        let array1 = ComputeResult.assumeSingleton <| inArray.DeviceArrays
        let result = // neighbour mapping logic: we create various copies of the array with offsets and length changes applied so that we can use map2, map3, etc. between them
            match neighbourSpec with
            |NeighbourMapping.ImmediateLeft code -> // neighbour mapping of X_i and X_(i-1)
                let array2 = ComputeArrays.createArrayOffset -1 None array1
                let result = ComputeResult.assumeSingleton <| mapN code [ResComputeArray array1; ResComputeArray array2]
                createArrayOrOffsetFromSpec mapLengthSpec 
                    (ComputeArrays.createArrayOffset 0 (Some <| array1.Length) result) // length preserving case
                    (ComputeArrays.createArrayOffset 1 (Some <| array1.Length-1) result) // shrinking case : 1 element shorter with 1 positive offset
            |NeighbourMapping.ImmediateRight code -> // neighbour mapping of X_i and X_(i+1)
                let array2 = ComputeArrays.createArrayOffset 1 None array1
                let result = ComputeResult.assumeSingleton <| mapN code [ResComputeArray array1; ResComputeArray array2]
                createArrayOrOffsetFromSpec mapLengthSpec 
                    (ComputeArrays.createArrayOffset 0 (Some <| array1.Length) result) // length preserving case
                    (ComputeArrays.createArrayOffset 0 (Some <| array1.Length-1) result) // shrinking case : 1 element shorter with 0 offset
            |NeighbourMapping.Stencil2 code -> // neighbour mapping of X_(i-1) and X_(i+1)
                let array2 = ComputeArrays.createArrayOffset -1 None array1
                let array3 = ComputeArrays.createArrayOffset 1 None array1
                let result = ComputeResult.assumeSingleton <| mapN code [ResComputeArray array2; ResComputeArray array3]
                createArrayOrOffsetFromSpec mapLengthSpec 
                    (ComputeArrays.createArrayOffset 0 (Some <| array1.Length) result) // length preserving case
                    (ComputeArrays.createArrayOffset 1 (Some <| array1.Length-2) result) // shrinking case : 2 elements shorter with 1 positive offset
            |NeighbourMapping.Stencil3 code -> // neighbour mapping of X_i, X_(i-1) and X_(i+1)
                let array2 = ComputeArrays.createArrayOffset -1 None array1
                let array3 = ComputeArrays.createArrayOffset 1 None array1
                let result = ComputeResult.assumeSingleton <| mapN code [ResComputeArray array1; ResComputeArray array2; ResComputeArray array3]
                createArrayOrOffsetFromSpec mapLengthSpec 
                    (ComputeArrays.createArrayOffset 0 (Some <| array1.Length) result) // length preserving case
                    (ComputeArrays.createArrayOffset 1 (Some <| array1.Length-2) result) // shrinking case : 2 elements shorter with 1 positive offset
            |NeighbourMapping.Stencil5 code -> // neighbour mapping of X_i, X_(i-2), X_(i-1), X_(i+1) and X_(i+2)
                let array2 = ComputeArrays.createArrayOffset -2 None array1
                let array3 = ComputeArrays.createArrayOffset -1 None array1
                let array4 = ComputeArrays.createArrayOffset 1 None array1
                let array5 = ComputeArrays.createArrayOffset 2 None array1
                let result = ComputeResult.assumeSingleton <| mapN code [ResComputeArray array1; ResComputeArray array2; ResComputeArray array3; ResComputeArray array4; ResComputeArray array5]
                createArrayOrOffsetFromSpec mapLengthSpec 
                    (ComputeArrays.createArrayOffset 0 (Some <| array1.Length) result) // length preserving case
                    (ComputeArrays.createArrayOffset 2 (Some <| array1.Length-4) result) // shrinking case : 4 elements shorter with 2 positive offset
        new devicearray<'b>(result) // convert typeless result to typed device array

    /// filters the array using a stable filter
    let filter (code : Expr<'a->devicebool>) (array : devicearray<'a>) =
        let performFilterMap lst = ComputeResult.assumeSingleton <| mapN code lst
        use result = performFilterMap [array.DeviceArrays]
        new devicearray<'a>(GeneralDeviceKernels.filterResult result array.DeviceArrays)

    /// partitions the array using a stable filter
    let partition (code : Expr<'a->devicebool>) (array : devicearray<'a>) =
        let performFilterMap lst = ComputeResult.assumeSingleton <| mapN code lst
        use result = performFilterMap [array.DeviceArrays]
        let trues, falses = GeneralDeviceKernels.partitionResult result array.DeviceArrays
        new devicearray<'a>(trues), new devicearray<'a>(falses)

    let evaluateMapsAndReconstructReduction (code : Expr<'a -> 'b -> 'a>) (array : devicearray<'b>) =
        let devArray = ComputeResult.assumeSingleton array.DeviceArrays
        match code with
        |ShapeLambda (var, expr) ->
            let foldResults = seperateReductionVariable var expr array
            match foldResults with
            |ReduceExpr (foldExpr, mapExrList, varList) ->
                let mapResults = mapExrList |> List.map (fun mapExpr -> 
                    let funWithLambda = reApplyLambdas expr [] mapExpr
                    mapN funWithLambda [ResComputeArray devArray])

                let acc = Expr.Lambda(varList |> List.head, foldExpr)
                let mapLambdas = (acc, varList |> List.tail) ||> List.fold (fun acc v -> Expr.Lambda(v, acc))
                (Expr.Lambda(var, mapLambdas), mapResults)
            |_ -> failwith "Error"
        |_ -> failwith "Error"

    /// Reduction functions

    /// General higher order associative reduce function
    let assocReduce<'a, 'b when 'b :> IGPUType and 'a :> IGPUType> (code : Expr< 'b -> 'a -> 'b>) array =
        /// Apply reduction to element 1 and element 2 and reduce the result to an array of half size
        let offsetMap (code : Expr< 'b -> 'a -> 'b>) (array1 : ComputeArray) (array2 : ComputeArray) =
            using (ComputeResult.assumeSingleton <| mapN code [ResComputeArray array1; ResComputeArray array2]) 
                (fun result -> GeneralDeviceKernels.reduceToEvenIndices result)
        /// Recursively apply the reduction to element X_i and X_i+1 and merge the results until only one element remains
        let rec assocReduceIntrnl (code : Expr< 'b -> 'a -> 'b>) (array : ComputeArray) =
            match (array.Length) with
            |0 -> 
                raise <| System.ArgumentException("array cannot be empty", "array")
            |1 ->
                array 
            |_ ->
                let array1 = ComputeArrays.createArrayOffset 0 (None) array
                let array2 = ComputeArrays.createArrayOffset 1 (None) array
                let newArr = offsetMap code array1 array2 // evaluate reduction on X_i and X_i+1
                array.Dispose()
                assocReduceIntrnl code newArr // repeat until array of size 1
        // apply maps all elements and merge the results
        let foldExpr, mapResults = evaluateMapsAndReconstructReduction code array
        match mapResults with
        |[devArray] ->
            let foldExpr = Expr< 'b -> 'a -> 'b>.Cast foldExpr
            let nDevArray = assocReduceIntrnl (foldExpr) (ComputeResult.assumeSingleton devArray)
            devArray.Dispose()
            nDevArray |> deviceelement<'b>
        |_ ->
            raise <| System.InvalidOperationException("Reduction operation ended in an invalid state.")

    
    

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
    static member inline ofArray array = 
        DeviceHostTransfer.copyArrayToDevice array
    // ----
    // MAPS
    // ----

    /// Builds a new array whose elements are the results of applying the given function to each element of the array.
    static member mapQuote (expr) =
        DeviceArrayOps.map expr
    /// Builds a new array whose elements are the results of applying the given function to each element of the array.
    static member map ([<ReflectedDefinition>] expr) =
        DeviceArrayOps.map expr
    /// Builds a new array whose elements are the results of applying the given function to each element of the array.
    static member map2 ([<ReflectedDefinition>] expr) =
        DeviceArrayOps.map2 expr
    /// Builds a new array whose elements are the results of applying the given function to each element of the array and a specified number of its neighbours
    static member mapNeighbours neighbourSpec mapLengthSpec array =
        DeviceArrayOps.mapNeighbours neighbourSpec mapLengthSpec array

    // -------
    // FILTERS
    // -------

    /// Returns a new array containing only the elements of the array for which the given predicate returns true.  This operation performs a stable filter, i.e. does not change the order
    /// of the elements.
    static member filter ([<ReflectedDefinition>] expr) =
        DeviceArrayOps.filter expr

    static member partition ([<ReflectedDefinition>] expr) =
        DeviceArrayOps.partition expr

    // ----------
    // REDUCTIONS
    // ----------

    /// Applies a function to each element of in the array and merging the results into an array of half size recursively until all elements in the array have been merged.
    /// Note: The function which merges the accumulator and the element MUST be associative or this function will produce unexpected results.
    static member associativeReduce ([<ReflectedDefinition()>] code : Expr<'b -> 'a -> 'b>) =
        DeviceArrayOps.assocReduce code

    /// Returns the sum of each element of the device array.
    static member inline sum array =
        DeviceArray.associativeReduce (+) array

    /// Returns the sum of the results generated by applying the function to each element of the device array.
    static member inline sumBy ([<ReflectedDefinition>] expr) =
        DeviceArray.associativeReduce (+) << DeviceArray.mapQuote expr

    // ---------
    // ZIP/UNZIP
    // ---------

    /// Combines the two arrays into an array of tuples with two elements.
    static member zip array1 = DeviceArrayOps.zip array1

    /// Combines the three arrays into an array of tuples with three elements.
    static member zip3 array1 = DeviceArrayOps.zip3 array1

    /// Splits an array of pairs into two arrays.
    static member unzip array = DeviceArrayOps.unzip array

    /// Splits an array of triples into three arrays.
    static member unzip3 array = DeviceArrayOps.unzip3 array    


type DeviceElement =
    // UTILITY
    // -------
    static member inline toHost element = 
        DeviceHostTransfer.copyElementToHost element

