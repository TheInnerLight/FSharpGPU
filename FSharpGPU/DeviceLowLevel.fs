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

open System
open System.Runtime.InteropServices

/// The exception that is thrown when a call to the device fails
exception DeviceException of string
/// The exception that is thrown when an attempt to access an illegal device memory address is made
exception DeviceIllegalAccessException of string
/// The exception that is thrown when there is not enough device memory available to perform the requested operation.
exception DeviceOutOfMemoryException of string
/// Functions for device utility functionality
module internal DeviceInterop =
    let checkCudaResponse func =
        let value = func
        match value with
        |0 -> ()
        |1 -> raise <| DeviceException "cudaErrorMissingConfiguration : The device function being invoked (usually via cudaLaunch()) was not previously configured via the cudaConfigureCall() function."
        |2 -> raise <| DeviceOutOfMemoryException "The API call failed because it was unable to allocate enough memory to perform the requested operation."
        |3 -> raise <| DeviceException "cudaErrorInitializationError : The API call failed because the CUDA driver and runtime could not be initialized."
        |4 -> raise <| DeviceException "cudaErrorLaunchFailure : An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory."
        |5 -> raise <| DeviceException "cudaErrorPriorLaunchFailure : This indicated that a previous kernel launch failed. This was previously used for device emulation of kernel launches."
        |6 -> raise <| DeviceException "cudaErrorLaunchTimeout : This indicates that the device kernel took too long to execute."
        |7 -> raise <| DeviceException "cudaErrorLaunchOutOfResources : This indicates that a launch did not occur because it did not have appropriate resources. This error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count."
        |8 -> raise <| DeviceException "cudaErrorInvalidDeviceFunction : The requested device function does not exist or is not compiled for the proper device architecture."
        |9 -> raise <| DeviceException "cudaErrorInvalidConfiguration :	This indicates that a kernel launch is requesting resources that can never be satisfied by the current device. Requesting more shared memory per block than the device supports will trigger this error, as will requesting too many threads or blocks."
        |10 -> raise <| DeviceException "cudaErrorInvalidDevice : This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device."
        |11 -> raise <| DeviceException "cudaErrorInvalidValue : This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values."
        |77 -> raise <| DeviceIllegalAccessException "An illegal device memory access was attempted.  This could be the result of attempting out of bounds memory access."
        |_ -> raise <| DeviceException (sprintf "cuda error code %i" value)