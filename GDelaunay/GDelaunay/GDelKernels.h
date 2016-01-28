/*
Author: Ashwin Nanjappa and Cao Thanh Tung
Filename: GDelKernels.h

Copyright (c) 2013, School of Computing, National University of Singapore. 
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the National University of Singapore nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission from the National University of Singapore. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

*/

////////////////////////////////////////////////////////////////////////////////
//                               Star Device Code
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "GDelCommon.h"

// Forward Declarations
struct KerInsertData;
struct KerPointData;
struct KerStarData;
struct KerTetraData;
struct PredicateInfo;

enum KernelMode
{
    KernelModeInvalid,

    // Read quad from grid
    CountPerThreadPairs,
    GrabPerThreadPairs,
};

/////////////////////////////////////////////////////////////////// Predicate //

enum DPredicateBounds
{
    Splitter,       /* = 2^ceiling(p / 2) + 1.  Used to split floats in half. */
    Epsilon,        /* = 2^(-p).  Used to estimate roundoff errors. */

    /* A set of coefficients used to calculate maximum roundoff errors.          */
    Resulterrbound,
    CcwerrboundA,
    CcwerrboundB,
    CcwerrboundC,
    O3derrboundA,
    O3derrboundB,
    O3derrboundC,
    IccerrboundA,
    IccerrboundB,
    IccerrboundC,
    IsperrboundA,
    IsperrboundB,
    IsperrboundC,

    DPredicateBoundNum  // Number of bounds in this enum
};

enum DPredicateSizes
{
    // Size of each array
    Temp96Size  = 96,
    Temp192Size = 192,
    Det384xSize = 384,
    Det384ySize = 384,
    Det384zSize = 384,
    DetxySize   = 768,
    AdetSize    = 1152,
    AbdetSize   = 2304,
    CdedetSize  = 3456,
    DeterSize   = 5760,

    // Total size
    PredicateTotalSize = 0
    + Temp96Size
    + Temp192Size
    + Det384xSize
    + Det384ySize
    + Det384zSize
    + DetxySize
    + AdetSize
    + AbdetSize
    + CdedetSize
    + DeterSize
};

/////////////////////////////////////////////////////////// Predicate kernels //

__global__ void kerMarkLowerHullTetra
(
PredicateInfo   predInfo,
KerPointData    pointData,
KerStarData     starData,
KerIntArray     tetraTriMap
)
;
__global__ void
__launch_bounds__( MAX_PRED_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
kerGetProofExact
(
PredicateInfo   predInfo,
KerPointData    pointData,
KerStarData     starData,
KerIntArray     drownedStarArr,
int*            drownedVertArr,
int*            proofStarArr,
int*            proofVertArr
)
;
__global__ void
kerGetProofFast
(
PredicateInfo   predInfo,
KerPointData    pointData,
KerStarData     starData,
KerIntArray     drownedStarArr,
int*            drownedVertArr,
int*            proofStarArr,
int*            proofVertArr
)
;
__global__ void 
__launch_bounds__( MAX_PRED_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
kerMakeInitialConeExact
(
PredicateInfo   predInfo,
KerPointData    pointData,
KerStarData     starData,
KerBeneathData  beneathData,
KerInsertData   worksetData
)
;
__global__ void 
__launch_bounds__( MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
kerMakeInitialConeFast
(
PredicateInfo   predInfo,
KerPointData    pointData,
KerStarData     starData,
KerBeneathData  beneathData,
KerInsertData   worksetData
)
;
__global__ void
__launch_bounds__( MAX_PRED_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
kerMarkBeneathTrianglesExact
(
PredicateInfo   predInfo,
KerPointData    pointData,
KerStarData     starData,
KerBeneathData  beneathData,
KerInsertData   insertData,
KerIntArray     activeTriPosArr,
KerShortArray   activeTriInsNumArr,
int             insIdx
)
;
__global__ void
__launch_bounds__( MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
kerMarkBeneathTrianglesFast
(
PredicateInfo   predInfo,
KerPointData    pointData,
KerStarData     starData,
KerBeneathData  beneathData,
KerInsertData   insertData,
KerIntArray     activeTriPosArr,
KerShortArray   activeTriInsNumArr,
int             insIdx
)
;

///////////////////////////////////////////////////////////////////// Kernels //

__global__ void kerAppendValueToKey
(
KerIntArray keyArr,
int*        valArr,
int         bitsPerIndex
)
;
__global__ void kerCheckStarConsistency
( 
KerStarData     starData,
KerBeneathData  beneathData
)
;
__global__ void kerComputeTriangleCount( KerStarData, KerIntArray );
__global__ void kerConvertMapToCount
(
KerIntArray inMap,
int*        countArr,
int         dataNum
)
;
__global__ void kerCopyInsertionToNewHistory
(
KerIntArray     insVertArr,
int*            insVertStarArr,
KerIntArray     insStarVertMap,
int*            oldHistStarVertMap,
int             oldHistVertNum,
int*            newVertArr,
int*            newVertStarArr,
int*            newStarVertMap
)
;
__global__ void kerCopyOldToNewHistory
(
KerHistoryData  historyData,    // Old history
int*            newVertArr,
int*            newVertStarArr,
int*            newStarVertMap
)
;
__global__ void kerCopyPumpedUsingAtomic
(
KerInsertData   insertData,
KerIntArray     pumpStarArr,
int*            pumpVertArr,
int*            toIdxArr
)
;
__global__ void kerCopyWorksets
(
KerInsertData   insertData,
KerIntArray     fromStarArr,
int*            fromVertArr,
int*            fromMap
)
;
__global__ void kerCountPerStarInsertions( KerStarData, KerInsertData );
__global__ void kerCountPointsOfStar( KerStarData );
__global__ void
kerRestoreDrownedPairs
(
KerIntArray keyArr,
int*        valArr,
int         bitsPerIndex
)
;
__global__ void
kerOrderDrownedPairs
(
KerIntArray keyArr,
int*        valArr,
int         bitsPerIndex
)
;
__global__ void kerGatherPumpedInsertions
(
KerIntArray fromMap,
KerIntArray fromStarArr,
int*        fromVertArr,
KerIntArray pumpStarArr,
int*        pumpMap,
KerIntArray outStarArr,
int*        outVertArr
)
;
__global__ void kerGetActiveTriInsCount
(
KerStarData     starData,
KerTriPosArray  activeTriPosArr,
KerShortArray   activeTriInsNumArr
)
;
__global__ void kerGetActiveTriPos
(
KerStarData     starData,
KerIntArray     activeStarArr,
KerIntArray     activeTriMap,
KerTriPosArray  activeTriPosArr,
KerShortArray   activeTriInsNumArr
)
;
__global__ void
kerMarkSubmergedInsertions
(
KerStarData      starData,
KerInsertData    insertData
)
;
__global__ void kerGetActiveTriCount( KerStarData, KerIntArray, KerIntArray, int, bool );
__global__ void kerGetActiveTriCount( KerStarData, KerIntArray, KerIntArray );
__global__ void kerGetPerTriangleInsertions
(
KerStarData     starData,
KerIntArray     facetMap,
KerIntArray     triStarArr,
int*            triVertArr,
int             allTriInsertNum // Total number (unbounded) of triangle insertions
)
;
__global__ void kerGetPerTriangleCount
(
KerStarData starData,
int*        insCountArr
)
;
__global__ void kerGrabTetrasFromStars
(
KerStarData     starData,
KerTetraData    tetraData,
KerIntArray     tetraTriMap,
int*            triTetraMap,
LocTriIndex*    tetraCloneTriArr
)
;
__global__ void kerInitPredicate( RealType* );
__global__ void kerMakeAllStarMap( KerIntArray, KerIntArray, int );
__global__ void kerMakeCloneFacets
(
KerStarData     starData,
KerIntArray     tetraTriMap,
int*            triTetraMap,
int*            facetStarArr,
int*            facetTriArr
)
;
__global__ void kerMakeOldToNewTriMap( KerStarData, int, KerIntArray, KerIntArray, KerIntArray, KerTriStatusArray );
__global__ void kerMakeMissingData( const int*, int, KerPointData, KerMissingData );
__global__ void
kerMarkDuplicateDrownedPairs
(
KerIntArray keyArr,
int*        valArr
)
;
__global__ void kerMarkDuplicates( KerIntArray, KerIntArray ); 
__global__ void kerMarkIfInsertionInHistory
(
KerHistoryData  historyData,
KerIntArray     starArr,
int*            vertArr,
int             starNum
)
;
__global__ void kerMarkReversePairs
(
KerIntArray keyArr,
int*        valArr
)
;
__global__ void kerNoteOwnerTriangles( KerStarData, KerIntArray );
__global__ void kerReadPairsFromGrid( const int*, int, KerInsertData, KernelMode );
__global__ void kerRemoveValueFromKey
(
KerIntArray keyArr,
int         bitsPerIndex
) 
;
__global__ void kerStitchPointToHole
(
KerStarData     starData,
KerBeneathData  beneathData,
KerInsertData   insertData,
KerIntArray     activeStarArr,
int             insIdx
)
;
__global__ void kerGetCloneTriInfo
(
KerStarData     starData,
KerIntArray     facetStarArr,
int*            facetTriArr,
int*            triTetraMap,
LocTriIndex*    tetraCloneTriArr
)
;

/////////////////////////////////////////////////////////////////// Functions //

template< typename T >
__forceinline__ __device__ void cuSwap( T& v0, T& v1 )
{
    const T tmp = v0;
    v0          = v1;
    v1          = tmp;

    return;
}

__forceinline__ __device__ int flipToNeg( int val )
{
    CudaAssert( ( val >= 0 ) && "Invalid value for negation!" );
    return ( - val - 1 );
}

__forceinline__ __device__ int flipToPos( int val )
{
    CudaAssert( ( val < 0 ) && "Invalid value for un-negation!" );
    return ( - val - 1 );
}

// Calculate number of triangles for set of starNum 2-spheres
// Sum of number of points in all 2-spheres is pointNum
__forceinline__ __host__ __device__ int get2SphereTriangleNum( int starNum, int pointNum )
{
    // From 2-sphere Euler we have ( t = 2n - 4 )
    return ( 2 * pointNum ) - ( starNum * 4 );
}

// Convert 3D integer coordinate to index
__forceinline__ __host__ __device__ int coordToIdx( int gridWidth, int3 coord )
{
    return ( ( coord.z * ( gridWidth * gridWidth ) ) + ( coord.y * gridWidth ) + coord.x );
}

__forceinline__ __device__ int getCurThreadIdx()
{
    const int threadsPerBlock   = blockDim.x;
    const int curThreadIdx      = ( blockIdx.x * threadsPerBlock ) + threadIdx.x;
    return curThreadIdx;
}

__forceinline__ __device__ int getThreadNum()
{
    const int blocksPerGrid     = gridDim.x;
    const int threadsPerBlock   = blockDim.x;
    const int threadNum         = blocksPerGrid * threadsPerBlock;
    return threadNum;
}

__forceinline__ __device__ PredicateInfo getCurThreadPredInfo( PredicateInfo predInfo )
{
    const int curPredDataIdx        = getCurThreadIdx() * PredicateTotalSize;
    RealType* curPredData           = &( predInfo._data[ curPredDataIdx ] );
    const PredicateInfo curPredInfo = { predInfo._consts, curPredData };
    return curPredInfo;
}

////////////////////////////////////////////////////////////////////////////////
