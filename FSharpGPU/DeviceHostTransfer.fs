namespace NovelFS.FSharpGPU


module DeviceHostTransfer =
    type DeviceArrayCreator = 
        |DeviceArrayCreator
        /// transfer float array to device
        static member (&!!!!>) (DeviceArrayCreator, arr : float[]) =
            let mutable cudaPtr = System.IntPtr(0)
            DeviceInterop.initialiseCUDADoubleArray(arr, Array.length arr, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
            new devicearray<devicefloat>(new ComputeArray(ComputeDataType.ComputeFloat, cudaPtr, Array.length arr, FullArray, UserGenerated))
        /// transfer bool array to device
        static member (&!!!!>) (DeviceArrayCreator, arr : bool[]) =
            let mutable cudaPtr = System.IntPtr(0)
            DeviceInterop.initialiseCUDABoolArray(arr, Array.length arr, &cudaPtr) |> DeviceInterop.cudaCallWithExceptionCheck
            new devicearray<devicefloat>(new ComputeArray(ComputeDataType.ComputeFloat, cudaPtr, Array.length arr, FullArray, UserGenerated))
        /// transfer device float array to host
        static member (<&!!!!) (DeviceArrayCreator, array : devicearray<devicefloat>) =
            let devArray = ComputeResult.assumeSingleton (array.DeviceArrays)
            match devArray.ArrayType with
            |ComputeDataType.ComputeFloat ->
                let hostArray = Array.zeroCreate<float> (devArray.Length)
                DeviceInterop.retrieveCUDADoubleArray(devArray.CudaPtr, devArray.Offset, hostArray, hostArray.Length) |> DeviceInterop.cudaCallWithExceptionCheck
                hostArray
            |_ -> failwith "Invalid type"
        /// transfer device bool array to host
        static member (<&!!!!) (DeviceArrayCreator, array : devicearray<devicebool>) =
            let devArray = ComputeResult.assumeSingleton (array.DeviceArrays)
            match devArray.ArrayType with
            |ComputeDataType.ComputeBool ->
                let hostArray = Array.zeroCreate<int> (devArray.Length)
                DeviceInterop.retrieveCUDABoolArray(devArray.CudaPtr, devArray.Offset, hostArray, hostArray.Length) |> DeviceInterop.cudaCallWithExceptionCheck
                hostArray |> Array.map (function |0 -> false; |_ -> true)
            |_ -> failwith "Invalid type"
        /// transfer device float element to host
        static member (<<!!!!) (DeviceArrayCreator, array : deviceelement<devicefloat>) =
            let devArray = array.DeviceArray
            match devArray.ArrayType with
            |ComputeDataType.ComputeFloat ->
                let hostArray = Array.zeroCreate<float> (devArray.Length)
                DeviceInterop.retrieveCUDADoubleArray(devArray.CudaPtr, devArray.Offset, hostArray, hostArray.Length) |> DeviceInterop.cudaCallWithExceptionCheck
                hostArray |> Array.head
            |_ -> failwith "Invalid type"
        /// transfer device bool element to host
        static member (<<!!!!) (DeviceArrayCreator, array : deviceelement<devicebool>) =
            let devArray = array.DeviceArray
            match devArray.ArrayType with
            |ComputeDataType.ComputeBool ->
                let hostArray = Array.zeroCreate<int> (devArray.Length)
                DeviceInterop.retrieveCUDABoolArray(devArray.CudaPtr, devArray.Offset, hostArray, hostArray.Length) |> DeviceInterop.cudaCallWithExceptionCheck
                hostArray |> Array.map (function |0 -> false; |_ -> true) |> Array.head
            |_ -> failwith "Invalid type"

    /// copy an array to the device
    let inline copyArrayToDevice arr = DeviceArrayCreator &!!!!> arr
    let inline copyArrayToHost arr = DeviceArrayCreator <&!!!! arr
    let inline copyElementToHost arr = DeviceArrayCreator <<!!!! arr