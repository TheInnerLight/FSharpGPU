/*This file is part of FSharpGPU.

	FSharpGPU is free software : you can redistribute it and / or modify
	it under the terms of the GNU Affero General Public License as
	published by the Free Software Foundation, either version 3 of the
	License, or(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
	GNU Affero General Public License for more details.

	You should have received a copy of the GNU Affero General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/* This software contains source code provided by NVIDIA Corporation. */

/*Copyright © 2015 Philip Curzon */

#pragma once

struct ScanBlockAllocation{
	__int32** g_scanBlockSums;
	size_t g_numEltsAllocated = 0;
	size_t g_numLevelsAllocated = 0;
};

void prescanArray(__int32 *outArray, __int32 *inArray, int numElements, ScanBlockAllocation sba);

ScanBlockAllocation preallocBlockSums(size_t maxNumElements);

void deallocBlockSums(ScanBlockAllocation sba);