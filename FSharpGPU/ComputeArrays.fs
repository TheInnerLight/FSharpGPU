﻿namespace NovelFS.FSharpGPU

module internal ComputeArrays =
    /// Create an offset array from a supplied array and the specified offset
    let createArrayOffset offS newLength (array : ComputeArray) =
        match newLength with
        |None ->
            new ComputeArray(array.ArrayType, array.CudaPtr, array.Length, OffsetSubarray(offS), AutoGenerated)
        |Some n ->
            new ComputeArray(array.ArrayType, array.CudaPtr, n, OffsetSubarray(offS), AutoGenerated)

