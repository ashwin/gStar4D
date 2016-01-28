/*
Author: Ashwin Nanjappa and Cao Thanh Tung
Filename: GDelKernels.cu

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

///////////////////////////////////////////////////////////////////// Headers //

#include "GDelKernels.h"
#include "Geometry.h"

//////////////////////////////////////////////////// Exclusive-Inclusive Scan //

#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)

inline __device__ int warpScanInclusive(int idata, volatile int *s_Data)
{
    int pos = 2 * threadIdx.x - (threadIdx.x & (WARP_SIZE - 1));
    s_Data[pos] = 0;
    pos += WARP_SIZE;
    s_Data[pos] = idata;

    s_Data[pos] += s_Data[pos - 1];
    s_Data[pos] += s_Data[pos - 2];
    s_Data[pos] += s_Data[pos - 4];
    s_Data[pos] += s_Data[pos - 8];
    s_Data[pos] += s_Data[pos - 16];

    return s_Data[pos];
}

inline __device__ int warpScanInclusive(int idata, int *s_Data, int size)
{
    int pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for(int offset = 1; offset < size; offset <<= 1)
        s_Data[pos] += s_Data[pos - offset];

    return s_Data[pos];
}

inline __device__ int warpScanExclusive(int idata, int *s_Data, int size)
{
    return warpScanInclusive(idata, s_Data, size) - idata;
}

inline __device__ int scan1Inclusive(int idata, int *s_Data, int size)
{
    // Bottom-level inclusive warp scan
    int warpResult = warpScanInclusive(idata, s_Data);

    // Save top elements of each warp for exclusive warp scan
    // sync to wait for warp scans to complete (because s_Data is being overwritten)
    __syncthreads();
    if( (threadIdx.x & (WARP_SIZE - 1)) == (WARP_SIZE - 1) )
        s_Data[threadIdx.x >> LOG2_WARP_SIZE] = warpResult;

    // wait for warp scans to complete
    __syncthreads();
    if( threadIdx.x < (size >> LOG2_WARP_SIZE) )
    {
        // grab top warp elements
        int val = s_Data[threadIdx.x];
        // calculate exclsive scan and write back to shared memory
        s_Data[threadIdx.x] = warpScanExclusive(val, s_Data, size >> LOG2_WARP_SIZE);
    }

    // return updated warp scans with exclusive scan results
    __syncthreads();
    return warpResult + s_Data[threadIdx.x >> LOG2_WARP_SIZE];
}

///////////////////////////////////////////////////////////////////// Kernels //

__global__ void kerMakeMissingData
(
const int*      grid,
int             gridWidth,
KerPointData    pointData,
KerMissingData  missingData
)
{
    // Iterate through points
    for ( int pointIdx = getCurThreadIdx(); pointIdx < missingData._num; pointIdx += getThreadNum() )
    {
        CudaAssert( ( pointIdx < pointData._num ) && "Invalid point index!" );

        // Pick leader from grid location of point
        const Point3& point = pointData._pointArr[ pointIdx ];
        const int3 ptLoc    = { ( int ) point._p[0], ( int ) point._p[1], ( int ) point._p[2] };
        const int gridIdx   = coordToIdx( gridWidth, ptLoc );
        const int leader    = grid[ gridIdx ];

        CudaAssert( ( leader >= 0 ) && ( leader < pointData._num ) && "Invalid leader has won grid voxel!" );

        // Write leader for this point
        missingData._leaderArr[ pointIdx ] = leader;
    }

    return;
}

// Do NOT change this to inline device function.
// Inline device function was found to be comparitively slower!
#define READ_GRID_VALUE( dGrid, gridWidth, loc, value )     \
    /* Outer layer */                                       \
    if (   ( loc.x == -1 ) || ( loc.x == gridWidth )        \
        || ( loc.y == -1 ) || ( loc.y == gridWidth )        \
        || ( loc.z == -1 ) || ( loc.z == gridWidth ) )      \
    {                                                       \
        value = Marker;                                     \
    }                                                       \
    else                                                    \
    /* Inner region */                                      \
    {                                                       \
        const int curIdx    = coordToIdx( gridWidth, loc ); \
        value               = dGrid[ curIdx ];              \
    }

__forceinline__ __device__ void grabPair
( 
KerInsertData   worksetData, 
int             aVal, 
int             bVal, 
int&            curPairIdx,
KernelMode      mode
)
{
    CudaAssert( aVal != bVal ); 

    if ( aVal == Marker )
    {
        return; 
    }

    if ( GrabPerThreadPairs == mode )
    {
        worksetData._vertStarArr[ curPairIdx ]  = ( aVal < bVal ) ? aVal : bVal; 
        worksetData._vertArr[ curPairIdx ]      = ( aVal < bVal ) ? bVal : aVal;
    }
    
    ++curPairIdx;

    return;
}

// Read pairs from grid, one thread per row
// Invoked twice:
// 1: Count tetra
// 2: Read tetra
__global__ void kerReadPairsFromGrid
(
const int*      dGrid,
int             gridWidth,
KerInsertData   worksetData, 
KernelMode      mode
)
{
    // 8 voxels and their Voronoi vertices
    const int Root  = 0; 
    const int Opp   = 7; 
    const int LinkNum                   = 6;
    const int LinkVertex[ LinkNum + 1 ] = { 6, 2, 3, 1, 5, 4, 6 }; 

    int3 loc =   ( blockIdx.x <= gridWidth )
                ? make_int3( threadIdx.x - 1, blockIdx.x - 1, -1 )
                : make_int3( gridWidth - 1, threadIdx.x - 1, -1 ); // Read row on other side of grid

    const int curThreadIdx  = getCurThreadIdx();
    const int pairIdxBeg    =   ( CountPerThreadPairs == mode )
                                ? 0
                                : worksetData._starVertMap[ curThreadIdx ];
    int curPairIdx          = pairIdxBeg;

    int vals[8];
    int valIdx = 0;

    ////
    // Read one plane (4 voxels) in this row
    ////

    for ( int iy = 0; iy <= 1; ++iy ) { for ( int ix = 0; ix <= 1; ++ix )
    {
        const int3 curLoc = make_int3( loc.x + ix, loc.y + iy, loc.z );
        
        READ_GRID_VALUE( dGrid, gridWidth, curLoc, vals[ valIdx ] );

        ++valIdx;
    } } 

    ////
    // Move along row, using plane (4 voxels) from last read
    ////

    // Move along row
    for ( ; loc.z < gridWidth; ++loc.z )
    {     
        valIdx = 4;

        // Read next plane (4 voxels) in this row
        for ( int iy = 0; iy <= 1; ++iy ) { for ( int ix = 0; ix <= 1; ++ix )
        {
            const int3 curLoc = make_int3( loc.x + ix, loc.y + iy, loc.z + 1 );

            READ_GRID_VALUE( dGrid, gridWidth, curLoc, vals[ valIdx ] );

            ++valIdx;
        } } 

        // We have 8 values of cube of width 2
        // Check the main diagonal of cube
        const int rootVal       = vals[ Root ]; 
        const int oppVal        = vals[ Opp ]; 
        const bool hasMarker    = ( rootVal == Marker ) || ( oppVal == Marker );
        
        if ( rootVal != oppVal ) 
        {
            // Check 6 link pairs
            bool hasQuad    = false; 
            int aVal        = vals[ LinkVertex[ 0 ] ]; 

            for ( int vi = 0; vi < LinkNum; ++vi )
            {
                const int bVal = vals[ LinkVertex[ vi + 1 ] ]; 

                if (    ( aVal != bVal ) 
                     && ( aVal != rootVal ) && ( aVal != oppVal )
                     && ( bVal != rootVal ) && ( bVal != oppVal ) )
                {
                    grabPair( worksetData, rootVal, aVal, curPairIdx, mode ); 
                    grabPair( worksetData, oppVal, aVal, curPairIdx, mode ); 
                    grabPair( worksetData, rootVal, bVal, curPairIdx, mode ); 
                    grabPair( worksetData, oppVal, bVal, curPairIdx, mode ); 
                    grabPair( worksetData, aVal, bVal, curPairIdx, mode ); 

                    hasQuad = true; 
                }

                aVal = bVal; 
            }
            
            if ( hasQuad && !hasMarker ) // Has a quad
            {
                grabPair( worksetData, rootVal, oppVal, curPairIdx, mode ); 
            }           
        }

        // Store plane for next row
        vals[ 0 ] = vals[ 4 ];
        vals[ 1 ] = vals[ 5 ];
        vals[ 2 ] = vals[ 6 ];
        vals[ 3 ] = vals[ 7 ]; 
    }

    ////
    // Write count of thread
    ////

    if ( CountPerThreadPairs == mode )
    {
        worksetData._vertArr[ curThreadIdx ] = curPairIdx - pairIdxBeg;
    }

    return;
}

__global__ void kerCopyWorksets
(
KerInsertData   insertData,
KerIntArray     fromStarArr,
int*            fromVertArr,
int*            fromMap
)
{
    // Iterate current worksets
    for ( int fromIdx = getCurThreadIdx(); fromIdx < fromStarArr._num; fromIdx += getThreadNum() )
    {
        const int star          = fromStarArr._arr[ fromIdx ];
        const int fromVertBeg   = fromMap[ star ];
        const int toVertBeg     = insertData._starVertMap[ star ];

        const int locIdx    = fromIdx - fromVertBeg;
        const int toIdx     = toVertBeg + locIdx;

        insertData._vertStarArr[ toIdx ]    = star;
        insertData._vertArr[ toIdx ]        = fromVertArr[ fromIdx ];
    }

    return;
}

__global__ void kerCopyPumpedUsingAtomic
(
KerInsertData   insertData,
KerIntArray     pumpStarArr,
int*            pumpVertArr,
int*            toIdxArr
)
{
    // Iterate pumped pairs
    for ( int fromIdx = getCurThreadIdx(); fromIdx < pumpStarArr._num; fromIdx += getThreadNum() )
    {
        const int star  = pumpStarArr._arr[ fromIdx ];
        const int toIdx = atomicAdd( &toIdxArr[ star ], 1 );

        insertData._vertStarArr[ toIdx ]    = star;
        insertData._vertArr[ toIdx ]        = pumpVertArr[ fromIdx ];
    }

    return;
}

__forceinline__ __device__ bool isVertexInRange
(
const int*  arr,
int         beg,
int         end,
int         key
)
{
    int idx = beg;

    while ( idx < end )
    {
        if ( arr[ idx ] == key )
        {
            return true;
        }

        ++idx;
    }

    return false;
}

__global__ void kerGatherPumpedInsertions
(
KerIntArray fromMap,
KerIntArray fromStarArr,
int*        fromVertArr,
KerIntArray actStarArr,
int*        actPumpMap,
KerIntArray outStarArr,
int*        outVertArr
)
{
    const int starNum       = fromMap._num;
    const int pumpStarNum   = actStarArr._num;

    // Iterate *only* stars needing pumping
    for ( int idx = getCurThreadIdx(); idx < pumpStarNum; idx += getThreadNum() )
    {
        const int star      = actStarArr._arr[ idx ];
        const int pumpBeg   = actPumpMap[ idx ];
        const int pumpEnd   = ( ( idx + 1 ) < pumpStarNum ) ? actPumpMap[ idx + 1 ] : outStarArr._num;
        int pumpIdx         = pumpBeg;

        CudaAssert( ( pumpEnd > pumpBeg ) && "Star with no pumping!" );

        ////
        // Pump starving stars using workset of their neighbours
        // Note: There will always be at least ONE item in starving workset
        ////

        const int fromBeg   = fromMap._arr[ star ];
        const int fromEnd   = ( ( star + 1 ) < starNum ) ? fromMap._arr[ star + 1 ] : fromStarArr._num;
        bool donePumping    = false;

        CudaAssert( ( fromEnd > fromBeg ) && "Star has at least *one* workset item!" );

        // Iterate workset of starving star
        for ( int fromIdx = fromBeg; fromIdx < fromEnd; ++fromIdx ) 
        {
            const int neiVert       = fromVertArr[ fromIdx ];
            const int neiVertBeg    = fromMap._arr[ neiVert ];
            const int neiVertEnd    = ( ( neiVert + 1 ) < starNum ) ? fromMap._arr[ neiVert + 1 ] : fromStarArr._num;

            // Iterate workset of neighbour
            for ( int candidateIdx = neiVertBeg; candidateIdx < neiVertEnd; ++candidateIdx )
            {
                const int candidateVert = fromVertArr[ candidateIdx ]; 

                if ( star == candidateVert )
                {
                    continue;
                }

                // Check if already there in workset
                if (    !isVertexInRange( fromVertArr, fromBeg, fromEnd, candidateVert )
                    &&  !isVertexInRange( outVertArr, pumpBeg, pumpIdx, candidateVert ) )
                {
                    // Add ordered insertion
                    outStarArr._arr[ pumpIdx ] = star;
                    outVertArr[ pumpIdx ]      = candidateVert;
                    ++pumpIdx;

                    // Check if pumping is enough
                    if ( pumpIdx == pumpEnd )
                    {
                        donePumping = true; 
                        break; 
                    }
                }
            }

            if ( donePumping )
            {
                break; 
            }
        }

        ////
        // Borrowing from neighbour is not enough. So, use 0,1,2... stars
        // Note: This actually happens for few points
        ////

        int starIdx = 0; 

        while ( pumpIdx < pumpEnd )
        {
            CudaAssert( ( starIdx < starNum ) && "Not enough points in the world to pump this star!" ); 

            if ( star == starIdx )
            {
                continue;
            }

            // Check if it's already there
            if (    !isVertexInRange( fromVertArr, fromBeg, fromEnd, starIdx )
                &&  !isVertexInRange( outVertArr, pumpBeg, pumpIdx, starIdx ) )
            {
                // Add ordered insertion
                outStarArr._arr[ pumpIdx ] = star;
                outVertArr[ pumpIdx ]      = starIdx;
                ++pumpIdx; 
            }

            ++starIdx; 
        }
    }

    return;
}

// Given a list of sorted numbers (has duplicates and is non-contiguous) create a map
// Note: The input map *should* already be initialized to -1 !!!
// Guarantees:
// (1) For a number that is in input list, its map value and its next number's map value will be correct
// (2) For a number that is not in input list, either its map value is -1, the next one is -1, or size is 0
__global__ void kerMakeAllStarMap
(
KerIntArray inArr,
KerIntArray allStarMap,
int         starNum
)
{
    const int curThreadIdx = getCurThreadIdx(); 

    // Iterate input list of numbers
    for ( int idx = curThreadIdx; idx < inArr._num; idx += getThreadNum() )
    {
        const int curVal    = inArr._arr[ idx ]; 
        const int nextVal   = ( ( idx + 1 ) < inArr._num ) ? inArr._arr[ idx + 1 ] : starNum - 1; 

        CudaAssert( ( curVal <= nextVal ) && "Input array of numbers is not sorted!" );

        // Number changes at this index
        if ( curVal != nextVal )
        {
            allStarMap._arr[ curVal + 1 ]   = idx + 1; 
            allStarMap._arr[ nextVal ]      = idx + 1;
        }
    }

    if ( ( 0 == curThreadIdx ) && ( inArr._num > 0 ) )
    {
        const int firstVal          = inArr._arr[ 0 ];
        allStarMap._arr[ firstVal ] = 0;    // Zero index for first value in input list
    }

    return;
}

__forceinline__ __device__ TriPositionEx getNextFreeTri
(
KerStarData     starData,
StarInfo        starInfo,
TriPositionEx&  freeTriPosEx,
int&            maxStarSize
) 
{
    do
    {
        // Increment free triangle location
        starInfo.moveToNextTri( freeTriPosEx );

        const TriangleStatus status = starData.triStatusAt( freeTriPosEx );

        if ( Free == status )
        {
            ////
            // Update maximum star size if needed
            ////

            const int locIdx = starInfo.toLocTriIdx( freeTriPosEx );

            if ( locIdx >= maxStarSize ) 
            {
                maxStarSize = locIdx + 1; 
            }

            return freeTriPosEx;
        }

    } while ( true );

    CudaAssert( false && "No free triangle found!" );

    return freeTriPosEx; 
}

// Find first valid triangle adjacent to hole boundary
// There must be one!
__device__ void findFirstHoleSegment
(
KerStarData     starData,
StarInfo        starInfo,
TriPositionEx   beneathTriPosEx,
TriPositionEx&  firstTriPosEx, 
int&            firstVi,
TriPositionEx&  firstHoleTriPosEx
)
{
    // Check the beneath triangle we know if it's on the hole boundary
    const TriangleOpp triOppFirst = starData.triOppAt( beneathTriPosEx );

    for ( int vi = 0; vi < 3; ++vi ) 
    {
        const TriPositionEx oppTriPosEx = starInfo.locToTriPosEx( triOppFirst.getOppTri(vi) );
        const TriangleStatus status     = starData.triStatusAt( oppTriPosEx );

        if ( Free != status )  // Found a hole edge
        {
            firstTriPosEx       = oppTriPosEx; 
            firstHoleTriPosEx   = beneathTriPosEx;
            firstVi             = triOppFirst.getOppVi( vi ); 

            return; 
        }                
    }

    // Iterate triangles
    for ( int locTriIdx = 0; locTriIdx < starInfo._locTriNum; ++locTriIdx )
    {
        const TriPositionEx triPosEx    = starInfo.locToTriPosEx( locTriIdx );
        const TriangleStatus status     = starData.triStatusAt( triPosEx );

        // Ignore non-beneath triangles
        if ( Free == status )
        {
            continue;
        }

        const TriangleOpp triOpp = starData.triOppAt( triPosEx );

        // Iterate segments of beneath triangle
        for ( int vi = 0; vi < 3; ++vi )
        {
            const TriPositionEx triOppPosEx = starInfo.locToTriPosEx( triOpp.getOppTri(vi) );
            const TriangleStatus status     = starData.triStatusAt( triOppPosEx );

            if ( Free == status )  // Found a hole edge
            {
                firstTriPosEx       = triPosEx; 
                firstVi             = vi; 
                firstHoleTriPosEx   = triOppPosEx;      

                return; 
            }                
        }
    }

    CudaAssert( false && "Not found any hole triangle" );

    return;
}

__global__ void kerStitchPointToHole
(
KerStarData     starData,
KerBeneathData  beneathData,
KerInsertData   insertData,
KerIntArray     activeStarArr,
int             insIdx
)
{
    CudaAssert( ( insIdx >= 0 ) && "Invalid insertion index!" );

    // Thread 0-0 resets exact triangle counter
    if ( ( 0 == threadIdx.x ) && ( 0 == blockIdx.x ) )
    {
        beneathData._flagArr[ ExactTriCount ] = 0;
    }

    // Iterate active stars
    for ( int idx = getCurThreadIdx(); idx < activeStarArr._num; idx += getThreadNum() )
    {
        const int star                  = activeStarArr._arr[ idx ];
        const TriPosition beneathTriPos = beneathData._beneathTriPosArr[ star ]; 

        // Check if no beneath triangle found
        if ( -1 == beneathTriPos )
        {
            continue; // Nothing to do, since point is inside star
        }

        // Reset since we have read it
        beneathData._beneathTriPosArr[ star ] = -1;

        ////
        // Get insertion vertex
        ////

        const int insBeg                    = insertData._starVertMap[ star ];
        const int insLoc                    = insBeg + insIdx;
        const int insVert                   = insertData._vertArr[ insLoc ];
        insertData._vertStarArr[ insLoc ]   = flipToNeg( star ); // Mark as successful insertion (not drowned)

        ////
        // Find first hole segment
        ////
        
        const TriPositionEx beneathTriPosEx = triPosToEx( beneathTriPos ); 
        const StarInfo starInfo             = starData.getStarInfo( star );
        int maxStarSize                     = starData._maxSizeArr[ star ]; 
        int firstVi                         = -1; 
        TriPositionEx firstTriPosEx;
        TriPositionEx firstNewTriPosEx;

        // Use the first hole triangle to store the first new triangle
        findFirstHoleSegment( starData, starInfo, beneathTriPosEx, firstTriPosEx, firstVi, firstNewTriPosEx );

        ////
        // First stitched triangle
        ////

        // Get the first two vertices of the hole
        TriPositionEx curTriPosEx   = firstTriPosEx; 
        const Triangle& curTri      = starData.triangleAt( curTriPosEx ); 
        const int firstVert         = curTri._v[ ( firstVi + 1 ) % 3 ]; 
        int curVi                   = ( firstVi + 2 ) % 3; 
        int curVert                 = curTri._v[ curVi ]; 

        // Stitch the first triangle
        const Triangle firstNewTri                  = { insVert, curVert, firstVert };
        starData.triangleAt( firstNewTriPosEx )     = firstNewTri;
        starData.triStatusAt( firstNewTriPosEx )    = NewValidAndUnchecked;

        // Adjancency with opposite triangle
        TriangleOpp& firstNewTriOpp = starData.triOppAt( firstNewTriPosEx ); 
        firstNewTriOpp.setOpp( 0, starInfo.toLocTriIdx( firstTriPosEx ), firstVi );
        TriangleOpp& firstTriOpp    = starData.triOppAt( firstTriPosEx );            
        firstTriOpp.setOpp( firstVi, starInfo.toLocTriIdx( firstNewTriPosEx ), 0 );

        ////
        // Walk around outside of hole, stitching rest of triangles
        ////

        TriPositionEx freeTriPosEx      = makeTriPosEx( 0, starInfo._begIdx0 - 1 ); // Start from begin of array
        TriPositionEx prevNewTriPosEx   = firstNewTriPosEx; 

        // Walk outside the hole in CW direction
        while ( curVert != firstVert ) 
        {
            // Check opposite triangle
            const TriangleOpp& curTriOpp        = starData.triOppAt( curTriPosEx );
            const TriPositionEx gloOppTriPosEx  = starInfo.locToTriPosEx( curTriOpp.getOppTri( ( curVi + 2 ) % 3 ) ); 
            const TriangleStatus status         = starData.triStatusAt( gloOppTriPosEx ); 

            // Triangle is outside the hole
            if ( ( Free != status ) && ( NewValidAndUnchecked != status ) )
            {
                // Continue moving
                const int oppVi = curTriOpp.getOppVi( ( curVi + 2 ) % 3 ); 
                curVi           = ( oppVi + 2 ) % 3;                
                curTriPosEx     = gloOppTriPosEx;
            }
            // Triangle is in hole
            else
            {
                const TriPositionEx newTriPosEx =   ( Free == status )
                                                ? gloOppTriPosEx  // Reuse hole triangle
                                                : getNextFreeTri( starData, starInfo, freeTriPosEx, maxStarSize );

                // Get the next vertex in the hole boundary
                const int oppVi         = ( curVi + 2 ) % 3;
                const Triangle& curTri  = starData.triangleAt( curTriPosEx );
                const int nextVert      = curTri._v[ ( curVi + 1 ) % 3 ]; 

                // New triangle
                const int locNewTriIdx  = starInfo.toLocTriIdx( newTriPosEx );
                const Triangle newTri   = { insVert, nextVert, curVert };

                // Adjancency with opposite triangle
                TriangleOpp& curTriOpp  = starData.triOppAt( curTriPosEx );            
                curTriOpp.setOpp( oppVi, locNewTriIdx, 0 );
                TriangleOpp& newTriOpp  = starData.triOppAt( newTriPosEx ); 
                newTriOpp.setOpp( 0, starInfo.toLocTriIdx( curTriPosEx ), oppVi );

                // Adjacency with previous new triangle
                TriangleOpp& prevTriOpp = starData.triOppAt( prevNewTriPosEx );
                prevTriOpp.setOpp( 2, locNewTriIdx, 1 ); 
                newTriOpp.setOpp( 1, starInfo.toLocTriIdx( prevNewTriPosEx ), 2 ); 
                
                // Last hole triangle
                if ( nextVert == firstVert )
                {
                    TriangleOpp& firstTriOpp    = starData.triOppAt( firstNewTriPosEx );
                    firstTriOpp.setOpp( 1, locNewTriIdx, 2 ); 
                    newTriOpp.setOpp( 2, starInfo.toLocTriIdx( firstNewTriPosEx ), 1 ); 
                }

                // Store new triangle data
                starData.triangleAt( newTriPosEx )  = newTri;
                starData.triStatusAt( newTriPosEx ) = NewValidAndUnchecked;
                
                // Check if this is not a beneath triangle, but fresh new triangle
                if ( Free != status )
                {
                    // Star needs to be set *only* for such triangles
                    // Note: Saves about 15ms for 1M points on GTX 580
                    starData.triStarAt( newTriPosEx ) = star;
                }

                // Prepare for next triangle
                prevNewTriPosEx = newTriPosEx; 

                // Move to the next vertex
                curVi   = ( curVi + 1 ) % 3; 
                curVert = nextVert; 
            }
        }

        // Update bounds of max triangle
        starData._maxSizeArr[ star ] = maxStarSize;
    }

    return;
}

__global__ void kerCountPointsOfStar( KerStarData starData )
{
    // Iterate through stars
    for ( int star = getCurThreadIdx(); star < starData._starNum; star += getThreadNum() )
    {
        // Check if any insertions for this star
        if ( 0 == starData._insCountArr[ star ] ) 
        {
            continue; 
        }

        const StarInfo starInfo = starData.getStarInfo( star );
        int validTriCount       = starInfo._locTriNum;

        for ( int locTriIdx = 0; locTriIdx < starInfo._locTriNum; ++locTriIdx )
        {
            const TriPositionEx triPosEx = starInfo.locToTriPosEx( locTriIdx );

            if ( Free == starData.triStatusAt( triPosEx ) )
            {
                --validTriCount;
            }
        }

        // Get point count
        CudaAssert( ( 0 == ( ( validTriCount + 4 ) % 2 ) ) && "2-sphere triangle count not divisible by 2!" );

        starData._pointNumArr[ star ] = ( validTriCount + 4 ) / 2;
    }

    return;
}

__global__ void kerGetPerTriangleCount
(
KerStarData starData,
int*        insCountArr
)
{
    // Iterate all triangles
    for ( int triIdx = getCurThreadIdx(); triIdx < starData._totalTriNum; triIdx += getThreadNum() )
    {
        const TriPositionEx triPosEx    = starData.globToTriPosEx( triIdx );
        const TriangleStatus triStatus  = starData.triStatusAt( triPosEx );
        int insertCount                 = 0;

        // Only unchecked triangles
        if ( ( ValidAndUnchecked == triStatus ) || ( NewValidAndUnchecked == triStatus ) )
        {
            const Triangle tri = starData.triangleAt( triPosEx );

            // Iterate triangle edges
            for ( int vi = 0; vi < 3; ++vi )
            {
                const int v0 = tri._v[ vi ];
                const int v1 = tri._v[ ( vi + 1 ) % 3 ];

                if ( v0 < v1 )
                {
                    ++insertCount;
                }
            }
        }

        // Write insertion count
        insCountArr[ triIdx ] = insertCount;
    }

    return;
}

// Note that insertions can be bounded
__global__ void kerGetPerTriangleInsertions
(
KerStarData     starData,
KerIntArray     facetMap,
KerIntArray     triStarArr,
int*            triVertArr,
int             allTriInsertNum // Total number (unbounded) of triangle insertions
)
{
    // Iterate triangles
    for ( int triIdx = getCurThreadIdx(); triIdx < starData._totalTriNum; triIdx += getThreadNum() )
    {
        ////
        // Triangle out of bound
        ////

        const int insBeg    = facetMap._arr[ triIdx ];
        const int insEnd    = ( ( triIdx + 1 ) < starData._totalTriNum )
                            ? facetMap._arr[ triIdx + 1 ]
                            : allTriInsertNum; // Total number of unbounded insertions

        // Check if beyond insertion bound
        if ( insBeg >= triStarArr._num ) 
        {
            break; // No more work for this *thread*
        }

        ////
        // Triangle generating no insertion
        ////

        if ( insEnd == insBeg )
        {
            continue;
        }

        ////
        // Triangle generating insertions
        ////

        const TriPositionEx triPosEx    = starData.globToTriPosEx( triIdx );
        const Triangle tri              = starData.triangleAt( triPosEx );
        int insIdx                      = insBeg;

        // Iterate triangle edges
        for ( int vi = 0; vi < 3; ++vi )
        {
            const int v0 = tri._v[ vi ];
            const int v1 = tri._v[ ( vi + 1 ) % 3 ];

            if ( v0 < v1 )
            {
                // Note: When triangle lies on FacetMax boundary,
                // this write will go beyond bound by 1. Assumption is that the
                // arrays' "actual" size is at least +1 more than triStarArr._num

                triStarArr._arr[ insIdx ]   = v0;
                triVertArr[ insIdx ]        = v1;

                ++insIdx;
            }
        }

        CudaAssert( insIdx == insEnd );

        // All triangles with insertions *except* the one on bound
        if ( insIdx <= triStarArr._num )
        {
            TriangleStatus& triStatus = starData.triStatusAt( triPosEx );

            CudaAssert( ( ( Free != triStatus ) && ( Valid != triStatus ) ) && "Triangle has insertions, so has to be unchecked!" );

            triStatus = Valid; // Reset triangle status
        }
        // Triangle on boundary
        else
        {
            //CudaAssert( false && "There is a triangle on boundary! Not an error, just an informational message!" );
        }
    }

    return;
}

__global__ void kerCountPerStarInsertions
(
KerStarData      starData,
KerInsertData    insertData
)
{
    for ( int star = getCurThreadIdx(); star < starData._starNum; star += getThreadNum() )
    {
        ////
        // Update point number to be inserted AND drowned number
        ////

        const int insBeg  = insertData._starVertMap[ star ];
        int insEnd        = ( star < ( starData._starNum - 1 ) ) ? insertData._starVertMap[ star + 1 ] : insertData._vertNum;

        if ( ( -1 == insBeg ) || ( -1 == insEnd ) ) 
        {
            insEnd = insBeg; 
        }

        const int insPointNum = insEnd - insBeg;

        CudaAssert( ( insPointNum >= 0 ) && "Invalid indices!" );

        // Insert point count for this star
        starData._insCountArr[ star ] = insPointNum;

        ////
        // Drowned count for this star
        // Given star of n link points and m insertion points, only a maximum
        // of (n + m - 4) points can drown
        ////

        const int starPointNum      = starData._pointNumArr[ star ];
        const int totalPointNum     = starPointNum + insPointNum;

        // Update star point count
        starData._pointNumArr[ star ] = totalPointNum;
    }

    return;
}

__global__ void kerCopyOldToNewHistory
(
KerHistoryData  historyData,    // Old history[1]
int*            newVertArr,
int*            newVertStarArr,
int*            newStarVertMap
)
{
    CudaAssert( ( historyData._vertNum[1] > 0 ) && "There is no history[1]! Use the array directly, no need for this kernel!" );

    // Iterate old history vertices
    for ( int curVertIdx = getCurThreadIdx(); curVertIdx < historyData._vertNum[1]; curVertIdx += getThreadNum() )
    {
        const int star          = historyData._vertStarArr[1][ curVertIdx ];
        const int vertBeg       = historyData._starVertMap[1][ star ];
        const int locVertIdx    = curVertIdx - vertBeg;
        const int newVertBeg    = newStarVertMap[ star ];
        const int newVertLoc    = newVertBeg + locVertIdx;

        CudaAssert( ( vertBeg >= 0 ) && "Vertex that exists cannot have invalid map value!" );
        CudaAssert( ( newVertBeg >= 0 ) && "Vertex that exists cannot have invalid map value!" );

        // Copy from old to new location
        newVertArr[ newVertLoc ]        = historyData._vertArr[1][ curVertIdx ];
        newVertStarArr[ newVertLoc ]    = star;
    }

    return;
}

__global__ void kerCopyInsertionToNewHistory
(
KerIntArray insVertArr,
int*        insVertStarArr,
KerIntArray insStarVertMap,
int*        oldHistStarVertMap,
int         oldHistVertNum,
int*        newVertArr,
int*        newVertStarArr,
int*        newStarVertMap
)
{
    const int starNum = insStarVertMap._num;

    // Iterate current insertions
    for ( int curInsIdx = getCurThreadIdx(); curInsIdx < insVertArr._num; curInsIdx += getThreadNum() )
    {
        ////
        // Location of *this* insertion vertex
        ////

        const int star      = insVertStarArr[ curInsIdx ];
        const int insBeg    = insStarVertMap._arr[ star ];
        const int insEnd    = ( ( star + 1 ) < starNum ) ? insStarVertMap._arr[ star + 1 ] : insVertArr._num;
        int insLocIdx       = curInsIdx - insBeg;

        CudaAssert( ( insBeg >= 0 ) && ( insEnd >= 0 ) && "*All* insertion map values must be valid!" );

        ////
        // Old history size of this star
        ////

        const int oldVertBeg    = oldHistStarVertMap[ star ];
        const int oldVertEnd    = ( ( star + 1 ) < starNum ) ? oldHistStarVertMap[ star + 1 ] : oldHistVertNum;
        const int oldVertNum    = oldVertEnd - oldVertBeg;

        CudaAssert( ( oldVertBeg >= 0 ) && ( oldVertEnd >= 0 ) && "*All* old history map values must be valid!" );

        ////
        // Destination of *this* insertion index
        ////

        const int newVertBeg    = newStarVertMap[ star ];
        const int newInsLoc     = newVertBeg + oldVertNum + insLocIdx;

        ////
        // Copy insertion to new history
        ////

        newVertArr[ newInsLoc ]     = insVertArr._arr[ curInsIdx ];
        newVertStarArr[ newInsLoc ] = star;
    }

    return;
}

__global__ void kerComputeTriangleCount
(
KerStarData starData,
KerIntArray triNumArr
)
{
    const float ExpandFactor = 1.0f;

    // Iterate through stars
    for ( int star = getCurThreadIdx(); star < starData._starNum; star += getThreadNum() )
    {
        // Current number of triangles
        const StarInfo starInfo = starData.getStarInfo( star );
        const int curTriNum     = starInfo._locTriNum;

        // Expected number of points (current + expected insertions)
        const int expPointNum = starData._pointNumArr[ star ];

        // Expected number of triangles
        const int insTriNum     = get2SphereTriangleNum( 1, expPointNum );
        const int newTriNum     = insTriNum * ExpandFactor;
        triNumArr._arr[ star ]  = max( newTriNum, curTriNum ) - starInfo._size0;    // Only "expand" second array
    }

    return;
}

__global__ void kerNoteOwnerTriangles
(
KerStarData starData,
KerIntArray tetraTriMap
)
{
    // Iterate triangles
    for ( int triIdx = getCurThreadIdx(); triIdx < starData._totalTriNum; triIdx += getThreadNum() )
    {
        const TriPositionEx triPosEx    = starData.globToTriPosEx( triIdx );
        const TriangleStatus status     = starData.triStatusAt( triPosEx ); 

        if ( Free == status ) 
        {
            continue; 
        }

        ////
        // Check if triangle's star is its owner
        ////

        const Triangle tri  = starData.triangleAt( triPosEx );
        const int star      = starData.triStarAt( triPosEx );

        if ( ( star < tri._v[0] ) & ( star < tri._v[1] ) & ( star < tri._v[2] ) )
        {
            tetraTriMap._arr[ triIdx ] = triIdx;
        }
    }

    return;
}

__global__ void kerGrabTetrasFromStars
(
KerStarData     starData,
KerTetraData    tetraData,
KerIntArray     tetraTriMap,
int*            triTetraMap,
LocTriIndex*    tetraCloneTriArr
)
{
    const int tetraNum = tetraTriMap._num;

    // Iterate all tetrahedrons
    for ( int tetIdx = getCurThreadIdx(); tetIdx < tetraNum; tetIdx += getThreadNum() )
    {
        // Construct 4 vertices of the tetra
        const int triIdx                = tetraTriMap._arr[ tetIdx ];
        const TriPositionEx triPosEx    = starData.globToTriPosEx( triIdx );
        const int fromStar              = starData.triStarAt( triPosEx ); 
        const StarInfo starInfo         = starData.getStarInfo( fromStar );
        const Triangle tri              = starData.triangleAt( triPosEx );

        Tetrahedron tetra;
        const int v0    = tri._v[0]; 
        tetra._v[0]     = v0;
        tetra._v[1]     = tri._v[1];
        tetra._v[2]     = tri._v[2];
        tetra._v[3]     = fromStar;

        ////
        // Set 3 opposites of this tetra
        ////

        const TriangleOpp triOpp = starData.triOppAt( triPosEx ); 

        for ( int vi = 0; vi < 3; ++vi ) 
        {
            const int gloOppTriIdx  = starInfo.toGlobalTriIdx( triOpp.getOppTri( vi ) );
            tetra._opp[ vi ]        = triTetraMap[ gloOppTriIdx ];
        }

        ////
        // Set 4th opposite of this tetra
        ////

        // Get clone triangle
        const StarInfo toStarInfo       = starData.getStarInfo( v0 );   
        const LocTriIndex locTriIdx     = tetraCloneTriArr[ tetIdx ];
        const TriPositionEx toTriPosEx  = toStarInfo.locToTriPosEx( locTriIdx );
        const Triangle toTri            = starData.triangleAt( toTriPosEx ); 

        // Find tetra opposite _v[ 3 ]
        const int fromStarIdx       = toTri.indexOfVert( fromStar ); 
        const TriangleOpp& toTriOpp = starData.triOppAt( toTriPosEx ); 
        const int starTriOppIdx     = toStarInfo.toGlobalTriIdx( toTriOpp.getOppTri( fromStarIdx ) ); 
        tetra._opp[ 3 ]             = triTetraMap[ starTriOppIdx ];

        // Write back tetra
        tetraData._arr[ tetIdx ] = tetra;
    }

    return;
}

__forceinline__ __device__ bool checkIfStarHasTriangle
(
KerStarData starData,
int         star,
Triangle    inTri
)
{
    const StarInfo starInfo = starData.getStarInfo( star );

    // Iterate triangles of star
    for ( int locTriIdx = 0; locTriIdx < starInfo._locTriNum; ++locTriIdx )
    {
        const TriPositionEx triPosEx    = starInfo.locToTriPosEx( locTriIdx ); 
        const TriangleStatus status     = starData.triStatusAt( triPosEx );
        
        // Ignore free triangles
        if ( Free == status )
        {
            continue;
        }

        // Check if triangle has vertex
        const Triangle tri = starData.triangleAt( triPosEx );

        if ( tri.hasVertex( inTri._v[0] ) & tri.hasVertex( inTri._v[1] ) & tri.hasVertex( inTri._v[2] ) )
        {
            return true;
        }
    }

    return false;
}

__global__ void kerCheckStarConsistency
( 
KerStarData     starData,
KerBeneathData  beneathData
)
{
    // Iterate all triangles
    for ( int triIdx = getCurThreadIdx(); triIdx < starData._totalTriNum; triIdx += getThreadNum() )
    {
        ////
        // Ignore free triangles
        ////

        const TriPositionEx triPosEx    = starData.globToTriPosEx( triIdx );
        TriangleStatus& triStatus       = starData.triStatusAt( triPosEx ); 

        if ( Free == triStatus )
        {
            continue; 
        }

        CudaAssert( Valid == triStatus ); 

        ////
        // Check if triangle is consistent
        ////

        const int star      = starData.triStarAt( triPosEx ); 
        const Triangle tri  = starData.triangleAt( triPosEx ); 

        // Check for this tetra in other 3 stars
        for ( int vi = 0; vi < 3; ++vi ) 
        {
            const int toStar        = tri._v[ vi ];
            const Triangle toTri    = { star, tri._v[ ( vi + 1 ) % 3 ], tri._v[ ( vi + 2 ) % 3 ] };

            if ( !checkIfStarHasTriangle( starData, toStar, toTri ) ) 
            {
                beneathData._flagArr[ ExactTriCount ] = 1; 
                break; 
            }
        }
    }

    return;
}

__global__ void kerAppendValueToKey
(
KerIntArray keyArr,
int*        valArr,
int         bitsPerIndex
) 
{
    const int bitsPerValue  = 31 - bitsPerIndex;
    const int ValMask       = 1 << bitsPerValue;

    // Iterate array
    for ( int idx = getCurThreadIdx(); idx < keyArr._num; idx += getThreadNum() )
    {
        const int key = keyArr._arr[ idx ]; 
        const int val = valArr[ idx ];

        CudaAssert( ( key >= 0 ) && "Invalid key!" );

        keyArr._arr[ idx ] = ( ( key << bitsPerValue ) | ( val & ( ValMask - 1 ) ) ); 
    }

    return;
}

__global__ void kerRemoveValueFromKey
(
KerIntArray keyArr,
int         bitsPerIndex
) 
{
    const int bitsPerValue = 31 - bitsPerIndex;

    // Iterate array
    for ( int idx = getCurThreadIdx(); idx < keyArr._num; idx += getThreadNum() )
    {
        const int keyvalue = keyArr._arr[ idx ]; 

        CudaAssert( ( keyvalue >= 0 ) && "Key-Value is invalid!" );

        keyArr._arr[ idx ] = ( keyvalue >> bitsPerValue ); 
    }

    return;
}

// Key: ...  3  3  3  3  3 3 ...
// Val: ...  5  5  5  5  5 5 ...
// Res: ... -6 -6 -6 -6 -6 5 ...
__global__ void kerMarkDuplicates( KerIntArray keyArr, KerIntArray valueArr )
{
    // Iterate key array
    for ( int idx = getCurThreadIdx(); idx < keyArr._num; idx += getThreadNum() )
    {
        const int key = keyArr._arr[ idx ]; 
        const int val = valueArr._arr[ idx ]; 

        int nextIdx = idx + 1; 

        // Move right until end of array
        while ( nextIdx < keyArr._num ) 
        {
            const int nextKey = keyArr._arr[ nextIdx ]; 

            // Check if next key-val pair same as current pair
            if ( nextKey != key )
            {
                break; // Get out!
            }

            // Now this pair and next pair are same

            const int nextVal = valueArr._arr[ nextIdx ]; 

            // Compare *this* value to next
            if (    ( val == nextVal )              // nextVal is +ve
                ||  ( val == ( - nextVal - 1 ) )    // nextVal is -ve
                )
            {
                valueArr._arr[ idx ] = flipToNeg( val );    // Negate *this* value, so this pair can be removed
                break; 
            }

            ++nextIdx; 
        }
    }

    return;
}

// Make map AND also update triStar and triStatus array
__global__ void kerMakeOldToNewTriMap
(
KerStarData         oldStarData,
int                 oldTriNum,
KerIntArray         newTriMap,
KerIntArray         oldNewMap,
KerIntArray         newTriStar,
KerTriStatusArray   newTriStatus
)
{
    // Iterate through triangles
    for ( int oldTriIdx = getCurThreadIdx(); oldTriIdx < oldTriNum; oldTriIdx += getThreadNum() )
    {
        ////
        // Skip copying free triangle information
        ////

        const TriangleStatus status = oldStarData._triStatusArr[1][ oldTriIdx ]; 

        if ( Free == status )
        {
            continue; 
        }

        ////
        // Make map
        ////

        const int starIdx   = oldStarData._triStarArr[1][ oldTriIdx ];  // Star
        const int oldTriBeg = oldStarData._starTriMap[1][ starIdx ];    // Old location begin
        const int newTriBeg = newTriMap._arr[ starIdx ];                // New location begin
        const int newTriIdx = oldTriIdx - oldTriBeg + newTriBeg;        // New location

        oldNewMap._arr[ oldTriIdx ]     = newTriIdx;
        newTriStar._arr[ newTriIdx ]    = starIdx;
        newTriStatus._arr[ newTriIdx ]  = status;
    }

    return;
}

__global__ void kerGetActiveTriCount
(
KerStarData starData,
KerIntArray activeStarArr,
KerIntArray activeTriCountArr,
int         insIdx,
bool        isActiveBoundTight
)
{
    // Iterate active stars
    for ( int idx = getCurThreadIdx(); idx < activeStarArr._num; idx += getThreadNum() )
    {
        const int star      = activeStarArr._arr[ idx ];
        const int insNum    = starData._insCountArr[ star ]; 
        int maxSize         = 0;

        if ( insIdx < insNum ) 
        {
            if ( isActiveBoundTight )
            {
                // Exact bound of used triangles in star (tight bound)
                maxSize = starData._maxSizeArr[ star ];
            }
            else
            {
                // Total number of triangles of star (loose bound)
                const StarInfo& starInfo    = starData.getStarInfo( star ); 
                maxSize                     = starInfo._locTriNum; 
            }
        }

        activeTriCountArr._arr[ idx ] = maxSize; 
    }
    
    return;
}

__global__ void kerGetActiveTriCount
(
KerStarData starData,
KerIntArray activeStarArr,
KerIntArray activeTriCountArr
)
{
    // Iterate stars
    for ( int idx = getCurThreadIdx(); idx < activeStarArr._num; idx += getThreadNum() )
    {
        const int star                  = activeStarArr._arr[ idx ];
        const StarInfo starInfo         = starData.getStarInfo( star );
        activeTriCountArr._arr[ idx ]   = starInfo._locTriNum; 
    }
    
    return;
}

__forceinline__ __device__ bool _checkIfStarHasVertex
(
KerStarData starData,
int         star,
int         inVert
)
{
    const StarInfo starInfo = starData.getStarInfo( star );

    // Iterate triangles of star
    for ( int locTriIdx = 0; locTriIdx < starInfo._locTriNum; ++locTriIdx )
    {
        const TriPositionEx triPosEx    = starInfo.locToTriPosEx( locTriIdx ); 
        const TriangleStatus status     = starData.triStatusAt( triPosEx );
        
        // Ignore free triangles
        if ( Free == status )
        {
            continue;
        }

        // Check if triangle has vertex
        const Triangle tri = starData.triangleAt( triPosEx );

        if ( tri.hasVertex( inVert ) )
        {
            return true;
        }
    }

    return false;
}

__global__ void
kerMarkSubmergedInsertions
(
KerStarData      starData,
KerInsertData    insertData
)
{
    // Iterate through insertions
    for ( int idx = getCurThreadIdx(); idx < insertData._vertNum; idx += getThreadNum() )
    {
        ////
        // Ignore drowned insertions
        // Note: These have positive star value
        ////

        const int star = insertData._vertStarArr[ idx ];

        if ( star >= 0 )
        {
            continue;
        }

        ////
        // Check if insertion is submerged
        ////

        const int destStar              = flipToPos( star );
        const int vert                  = insertData._vertArr[ idx ];
        const bool starHasVert          = _checkIfStarHasVertex( starData, destStar, vert );
        insertData._vertStarArr[ idx ]  = starHasVert
                                        ? -1        // Successful insertion
                                        : destStar; // Submerged (is passed as drowned, so that it can get a proof)
    }

    return;
}

// Mark keys as -1 if ( val < key )
__global__ void kerMarkReversePairs
(
KerIntArray keyArr,
int*        valArr
)
{
    // Iterate items
    for ( int idx = getCurThreadIdx(); idx < keyArr._num; idx += getThreadNum() )
    {
        const int key = keyArr._arr[ idx ];
        const int val = valArr[ idx ];

        CudaAssert( ( key != val ) && "Invalid key-val pair in array!" );

        if ( val < key )
        {
            keyArr._arr[ idx ] = -1;
        }
    }

    return;
}

__global__ void kerGetCloneTriInfo
(
KerStarData     starData,
KerIntArray     facetStarArr,
int*            facetTriArr,
int*            triTetraMap,
LocTriIndex*    tetraCloneTriArr
)
{
    // Iterate opp-tetra facet items
    for ( int facetIdx = getCurThreadIdx(); facetIdx < facetStarArr._num; facetIdx += getThreadNum() )
    {
        // From- data
        const int fromTriIdx                = facetTriArr[ facetIdx ]; 
        const TriPositionEx fromTriPosEx    = starData.globToTriPosEx( fromTriIdx );
        const Triangle& fromTri             = starData.triangleAt( fromTriPosEx );
        const int fromStar                  = starData.triStarAt( fromTriPosEx ); 
        const int v0                        = fromTri._v[ 1 ]; 
        const int v1                        = fromTri._v[ 2 ]; 

        // To- data
        const int toStar            = facetStarArr._arr[ facetIdx ];
        const StarInfo toStarInfo   = starData.getStarInfo( toStar );

        ////
        // Check if to-star has from-triangle
        ////

        int foundLocIdx = -1;

        // Iterate triangles of to-star
        for ( int locTriIdx = 0; locTriIdx < toStarInfo._locTriNum; ++locTriIdx )
        {
            const TriPositionEx triPosEx    = toStarInfo.locToTriPosEx( locTriIdx );
            const Triangle toTri            = starData.triangleAt( triPosEx );

            // Check if triangle matches
            if ( toTri.hasVertex( fromStar ) & toTri.hasVertex( v0 ) & toTri.hasVertex( v1 ) )
            {
                const TriangleStatus status = starData.triStatusAt( triPosEx );

                // Matched triangle *might* be free, check for it
                // Highly unlikely, but stranger things have happened in this world!
                if ( Free == status )
                {
                    continue;
                }

                // Matches!
                foundLocIdx = locTriIdx;
                break;
            }
        }

        CudaAssert( ( -1 != foundLocIdx ) && "Triangle not found in to-star!" );

        ////
        // Set triIdx of tetra's clone AND point clone triangle to tetra
        // Note: This is non-cohesive read-write. Sorting and then cohesively writing was
        // tried and found to be a bit slower than this.
        ////

        const int ownTetIdx             = triTetraMap[ fromTriIdx ];    CudaAssert( ownTetIdx >= 0 );
        tetraCloneTriArr[ ownTetIdx ]   = foundLocIdx;

        const int toTriIdx      = toStarInfo.toGlobalTriIdx( foundLocIdx );
        triTetraMap[ toTriIdx ] = ownTetIdx;
    }

    return;
}

__forceinline__ __device__ bool isPairInHistory
(
KerHistoryData  historyData,
int             starNum,
int             v0,
int             v1
)
{
    CudaAssert( ( v0 >= 0 ) && ( v0 < starNum ) );

    // Iterate the two history arrays
    for ( int hi = 0; hi < 2; ++hi )
    {
        const int histBeg   = historyData._starVertMap[ hi ][ v0 ];
        const int histEnd   = ( ( v0 + 1 ) < starNum ) ? historyData._starVertMap[ hi ][ v0 + 1 ] : historyData._vertNum[ hi ];

        CudaAssert( ( histBeg >= 0 ) && ( histEnd >= 0 ) && "*All* history map values must be valid!" );

        // Search history for insertion
        for ( int histIdx = histBeg; histIdx < histEnd; ++histIdx )
        {
            if ( v1 == historyData._vertArr[ hi ][ histIdx ] )
            {
                return true;
            }
        }
    }

    return false;
}

__global__ void kerMarkIfInsertionInHistory
(
KerHistoryData  historyData,
KerIntArray     starArr,
int*            vertArr,
int             starNum
)
{
    // Iterate insertions
    for ( int idx = getCurThreadIdx(); idx < starArr._num; idx += getThreadNum() )
    {
        const int star  = starArr._arr[ idx ];
        const int vert  = vertArr[ idx ];

        if ( isPairInHistory( historyData, starNum, star, vert ) )
        {
            vertArr[ idx ] = -1;
        }
    }

    return;
}

__forceinline__ __device__ int getValidMapVal
(
int* arr,
int  inIdx
)
{
    // Check if simple map value

    int val = arr[ inIdx ];

    if ( -1 != val )
    {
        return val;
    }

    // Decrement back until we hit valid map value

    int idx = inIdx;

    while ( idx > 0 )
    {
        const int prevVal = arr[ --idx ];

        if ( -1 != prevVal )
        {
            // Write back so it can be found easier by next search
            arr[ inIdx ] = prevVal;

            return prevVal;
        }
    }

    // Write back so it can be found easier by next search
    arr[ 0 ]        = 0;
    arr[ inIdx ]    = 0;
    return 0;
}

__global__ void kerConvertMapToCount
(
KerIntArray inMap,
int*        countArr,
int         dataNum
)
{
    const int starNum = inMap._num;

    // Iterate array
    for ( int idx = getCurThreadIdx(); idx < starNum; idx += getThreadNum() )
    {
        const int beg   = inMap._arr[ idx ];
        int end         = ( ( idx + 1 ) < starNum ) ? inMap._arr[ idx + 1 ] : dataNum;

        if ( ( -1 == beg ) || ( -1 == end ) )
        {
            end = beg;
        }

        countArr[ idx ] = ( end - beg );
    }

    return;
}

__global__ void kerGetActiveTriPos
(
KerStarData     starData,
KerIntArray     activeStarArr,
KerIntArray     activeTriMap,
KerTriPosArray  activeTriPosArr,
KerShortArray   activeTriInsNumArr
)
{
    // Iterate active stars
    for ( int idx = getCurThreadIdx(); idx < activeStarArr._num; idx += getThreadNum() )
    {
        // Get triangle map of star
        const int triIdxBeg = activeTriMap._arr[ idx ]; 
        const int triIdxEnd = ( idx + 1 < activeStarArr._num ) ? activeTriMap._arr[ idx + 1 ] : activeTriPosArr._num;

        // No active triangles in star
        if ( triIdxBeg == triIdxEnd )
        {
            continue; 
        }

        const int star          = activeStarArr._arr[ idx ]; 
        const StarInfo starInfo = starData.getStarInfo( star ); 
  
        int locTriIdx = 0; 

        // First triangle of star get the proper info
        activeTriPosArr._arr[ triIdxBeg + locTriIdx ]       = ( starInfo._begIdx0 << 1 );
        activeTriInsNumArr._arr[ triIdxBeg + locTriIdx ]    = starData._insCountArr[ star ];
        ++locTriIdx; 

        const int size = min( triIdxEnd - triIdxBeg, starInfo._size0 ); 

        ////
        // Write as int4 for better performance
        ////

        for ( ; ( locTriIdx < size ) && ( ( ( triIdxBeg + locTriIdx ) & 3 ) > 0 ); ++locTriIdx )
        {
            activeTriPosArr._arr[ triIdxBeg + locTriIdx ] = - locTriIdx;
        }

        for ( ; locTriIdx + 3 < size; locTriIdx += 4 )
        {
            ( ( int4* ) activeTriPosArr._arr )[ ( triIdxBeg + locTriIdx ) >> 2 ] = make_int4( -locTriIdx, -locTriIdx-1, -locTriIdx-2, -locTriIdx-3 );
        }

        for ( ; locTriIdx < size; ++locTriIdx )
        {
            activeTriPosArr._arr[ triIdxBeg + locTriIdx ] = - locTriIdx;
        }

        if ( ( triIdxBeg + locTriIdx ) < triIdxEnd )
        {
            // First triangle of the star in the second array also get the proper info
            TriPosition triPos = ( ( starInfo._begIdx1MinusSize0 + starInfo._size0 ) << 1 ) + 1 ; 
            activeTriPosArr._arr[ triIdxBeg + locTriIdx ]       = triPos; 
            activeTriInsNumArr._arr[ triIdxBeg + locTriIdx ]    = starData._insCountArr[ star ];

            ++locTriIdx; 

            // Iterate non-first triangles of star, give them local index (flipped)
            for ( ; locTriIdx < triIdxEnd - triIdxBeg; ++locTriIdx )
                activeTriPosArr._arr[ triIdxBeg + locTriIdx ]  = - ( locTriIdx - starInfo._size0 );
        }
    }

    return;
}

__global__ void kerGetActiveTriInsCount
(
KerStarData     starData,
KerTriPosArray  activeTriPosArr,
KerShortArray   activeTriInsNumArr
)
{
    // Iterate active triangles
    for ( int idx = getCurThreadIdx(); idx < activeTriPosArr._num; idx += getThreadNum() )
    {
        TriPosition triFirstPos = activeTriPosArr._arr[ idx ];

        // Not first triangle of star
        if ( triFirstPos < 0 ) 
        {
            const int locTriIdx             = - triFirstPos; 
            activeTriInsNumArr._arr[ idx ]  = activeTriInsNumArr._arr[ idx - locTriIdx ]; 
            activeTriPosArr._arr[ idx ]     = activeTriPosArr._arr[ idx - locTriIdx ] + ( locTriIdx << 1 );
        }
    }

    return;
}

// Check if star-vert pair is ordered
// If not, order it and set a bit in vert to 1 (encoding)
__global__ void
kerOrderDrownedPairs
(
KerIntArray keyArr,
int*        valArr,
int         bitsPerIndex
)
{
    const int bitsPerValue  = 31 - bitsPerIndex;
    const int ValMask       = 1 << bitsPerValue;

    for ( int idx = getCurThreadIdx(); idx < keyArr._num; idx += getThreadNum() )
    {
        int key     = keyArr._arr[ idx ]; 
        int val     = valArr[ idx ];
        int flipped = 0;

        CudaAssert( ( key >= 0 ) && "Invalid key!" );

        // Order pair so that (key < val)
        if ( key > val )
        {
            cuSwap( key, val );
            flipped = 1;
        }

        // Write back
        keyArr._arr[ idx ]  = ( ( key << bitsPerValue ) | ( val & ( ValMask - 1 ) ) );  // Append val to key
        valArr[ idx ]       = ( val << 1 ) | flipped;                                   // Encode flipped status in val
    }

    return;
}

// Remove the encoding and put back original order
__global__ void
kerRestoreDrownedPairs
(
KerIntArray keyArr,
int*        valArr,
int         bitsPerIndex
)
{
    const int bitsPerValue = 31 - bitsPerIndex;

    for ( int idx = getCurThreadIdx(); idx < keyArr._num; idx += getThreadNum() )
    {
        int key             = keyArr._arr[ idx ];
        int val             = valArr[ idx ];
        const int flipped   = ( val & 1 );

        CudaAssert( ( key >= 0 ) && "Appended Key-Value is invalid!" );

        // Restore key and val
        key = ( key >> bitsPerValue );
        val = ( val >> 1 );

        // Restore original order
        if ( 1 == flipped )
        {
            cuSwap( key, val );
        }

        // Write back original pair
        keyArr._arr[ idx ]  = key;
        valArr[ idx ]       = val;
    }

    return;
}

__global__ void
kerMarkDuplicateDrownedPairs
(
KerIntArray keyArr,
int*        valArr
)
{
    for ( int idx = getCurThreadIdx(); idx < keyArr._num; idx += getThreadNum() )
    {
        if ( 0 == idx )
        {
            continue;
        }

        const int prevKey   = keyArr._arr[ idx - 1 ];
        const int curKey    = keyArr._arr[ idx ];

        // Duplicate pair, only *one* such drowned pair possible
        if ( prevKey == curKey )
        {
            valArr[ idx - 1 ]   = -1;
            valArr[ idx ]       = -1;
        }
    }

    return;
}

////////////////////////////////////////////////////////////////////////////////
