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

module internal CommonExpressions =
    let cpuToFsharpGPUVar (varTable : Map<Var,_>) expr =
        match expr with
        |Double f -> ResComputeFloat f
        |Bool b -> ResComputeBool b
        |Var(var) -> varTable.[var]
        |_ -> raise <| System.InvalidOperationException("unable to map cpu variable to gpu")

        
