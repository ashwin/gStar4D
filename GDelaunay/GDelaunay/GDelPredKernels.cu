/*
Author: Ashwin Nanjappa and Cao Thanh Tung
Filename: GDelPredKernels.cu

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
#include "GDelPredDevice.h"

///////////////////////////////////////////////////////////////////// Kernels //

template < bool doFast >
__forceinline__ __device__
void makeInitialCone
(
PredicateInfo   curPredInfo,
KerPointData    pointData,
KerStarData     starData,
KerBeneathData  beneathData,
KerInsertData   insertData,
int             star
)
{
    // For pentachoron of the orientation 0123v, the following are
    // the 4 link-triangles orientation as seen from v.
    // Opposite triangle indices are also the same!
    const int LinkTri[4][3] = {
        { 1, 2, 3 },
        { 0, 3, 2 },
        { 0, 1, 3 },
        { 0, 2, 1 },    };

    const int OppVi[4][3] = {
        { 0, 0, 0 },
        { 0, 2, 1 },
        { 1, 2, 1 },
        { 2, 2, 1 },    }; 

    ////
    // Initialize star-triangle map and other data
    ////

    const int vertBeg       = insertData._starVertMap[ star ];
    const int nextVertBeg   = ( star < ( starData._starNum - 1 ) ) ? insertData._starVertMap[ star + 1 ] : insertData._vertNum;
    const int vertNum       = nextVertBeg - vertBeg;

    CudaAssert( ( vertNum >= 4 ) && "Working set too small to create a cone!" );

    const int triIdxBeg = get2SphereTriangleNum( star, vertBeg );

    if ( doFast )
    {
        starData._starTriMap[0][ star ] = triIdxBeg;    
        starData._starTriMap[1][ star ] = 0;
        starData._maxSizeArr[ star ]    = 4; // 4 triangles in beginning
    }

    ////
    // Read 4 points to form cone
    ////

    int linkVert[4];
    for ( int pi = 0; pi < 4; ++pi )
    {
        linkVert[ pi ] = insertData._vertArr[ vertBeg + pi ];

        if ( doFast )
        {
            insertData._vertStarArr[ vertBeg + pi ] = flipToNeg( star ); // Mark insertion as successful
        }
    }

    ////
    // Form 4-simplex with 4 points and star point
    ////

    // Orientation
    const Orient ord    = doFast
                        ? orientation4Fast( curPredInfo, pointData, linkVert[0], linkVert[1], linkVert[2], linkVert[3], star )
                        : orientation4SoS( curPredInfo, pointData, linkVert[0], linkVert[1], linkVert[2], linkVert[3], star );

    if ( doFast && ( OrientZero == ord ) )
    {
        // Need exact check
        const int exactListIdx                      = atomicAdd( &beneathData._flagArr[ ExactTriCount ], 1 ); 
        beneathData._exactTriPosArr[ exactListIdx ] = star; // Meant for triPos, but we use it for storing star during initial cone creation

        return; // Get out!
    }

    CudaAssert( ( OrientZero != ord ) && "Orientation is zero!" );

    // Swap for -ve order
    if ( OrientNeg == ord )
    {
        cuSwap( linkVert[0], linkVert[1] );
    }

    ////
    // Write 4 triangles of 4-simplex
    ////

    for ( int ti = 0; ti < 4; ++ti )
    {
        Triangle tri;
        TriangleOpp triOpp;

        for ( int vi = 0; vi < 3; ++vi )
        {
            tri._v[ vi ]        = linkVert[ LinkTri[ ti ][ vi ] ];
            triOpp.setOpp( vi, LinkTri[ ti ][ vi ], OppVi[ ti ][ vi ] );
        }

        CudaAssert( ( star != tri._v[ 0 ] ) && ( star != tri._v[ 1 ] ) && ( star != tri._v[ 2 ] )
                    && "Star vertex same as one of its cone vertices!" ); 

        const TriPositionEx triPosEx        = makeTriPosEx( 0, triIdxBeg + ti );
        starData.triangleAt( triPosEx )     = tri;
        starData.triOppAt( triPosEx )       = triOpp;
        starData.triStarAt( triPosEx )      = star;
        starData.triStatusAt( triPosEx )    = ValidAndUnchecked;
    }

    return;
}

__global__ void 
__launch_bounds__( MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
kerMakeInitialConeFast
(
PredicateInfo   predInfo,
KerPointData    pointData,
KerStarData     starData,
KerBeneathData  beneathData,
KerInsertData   insertData
)
{
    const PredicateInfo curPredInfo = getCurThreadPredInfo( predInfo );

    // Iterate through stars
    for ( int star = getCurThreadIdx(); star < starData._starNum; star += getThreadNum() )
    {
        makeInitialCone< true >(
            curPredInfo,
            pointData,
            starData,
            beneathData,
            insertData,
            star );
    }

    return;
}

__global__ void 
__launch_bounds__( MAX_PRED_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP )
kerMakeInitialConeExact
(
PredicateInfo   predInfo,
KerPointData    pointData,
KerStarData     starData,
KerBeneathData  beneathData,
KerInsertData   insertData
)
{
    ////
    // Check if *any* exact check needed
    ////

    const int exactVertNum = beneathData._flagArr[ ExactTriCount ]; 

    if ( 0 == exactVertNum )
    {
        return; // No exact checks needed
    }

    ////
    // Do exact check
    ////

    const PredicateInfo curPredInfo = getCurThreadPredInfo( predInfo );

    // Iterate stars needing exact check
    for ( int idx = getCurThreadIdx(); idx < exactVertNum; idx += getThreadNum() )
    {
        const int star = beneathData._exactTriPosArr[ idx ]; // Stored by fast check

        makeInitialCone< false >(
            curPredInfo,
            pointData,
            starData,
            beneathData,
            insertData,
            star );
    }

    return;
}

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
{
    CudaAssert( ( insIdx >= 0 ) && "Invalid insertion index!" );

    const PredicateInfo curThreadPredInfo = getCurThreadPredInfo( predInfo );

    // Iterate active triangles
    for ( int idx = getCurThreadIdx(); idx < activeTriPosArr._num; idx += getThreadNum() )
    {
        // Check if any insertion for triangle
        if ( activeTriInsNumArr._arr[ idx ] <= insIdx )
        {
            continue;
        }

        // Read triangle position and status
        const TriPosition triPos        = activeTriPosArr._arr[ idx ]; 
        const TriPositionEx triPosEx    = triPosToEx( triPos );
        TriangleStatus& triStatus       = starData.triStatusAt( triPosEx ); 

        // Ignore free triangle
        if ( Free == triStatus )
        {
            continue;
        }
        
        CudaAssert(     ( ( Valid == triStatus ) || ( ValidAndUnchecked == triStatus ) || ( NewValidAndUnchecked == triStatus ) )
                    &&  "Invalid triangle status for fast-exact check!" );

        ////
        // Get insertion point
        ////

        const int star      = starData.triStarAt( triPosEx );
        const int insBeg    = insertData._starVertMap[ star ];
        const int insEnd    = ( star < ( starData._starNum - 1 ) ) ? insertData._starVertMap[ star + 1 ] : insertData._vertNum;
        const int insLoc    = insBeg + insIdx;

        CudaAssert( insLoc < insEnd );

        ////
        // Check if triangle beneath point
        ////

        const Triangle tri  = starData.triangleAt( triPosEx );
        const int insVert   = insertData._vertArr[ insLoc ];
        const Orient ord    = orientation4Fast( curThreadPredInfo, pointData, tri._v[0], tri._v[1], tri._v[2], star, insVert );

        // Needs exact predicate
        if ( OrientZero == ord )
        {
            // Embed original status in exact status
            triStatus = ( Valid == triStatus ) ? DoExactOnValid : DoExactOnUnchecked;

            ////
            // Store triangle position for later exact predicate
            // Note: Done only if (< ExactTriangleMax) triangles requiring exact check
            ////

            const int exactListIdx = atomicAdd( &beneathData._flagArr[ ExactTriCount ], 1 ); 

            if ( exactListIdx < ExactTriangleMax ) 
            {
                beneathData._exactTriPosArr[ exactListIdx ] = exToTriPos( triPosEx );
            }
        }
        // Triangle is beneath insertion point
        else if ( OrientNeg == ord )
        {
            beneathData._beneathTriPosArr[ star ]   = exToTriPos( triPosEx ); // Store beneath triangle position
            triStatus                               = Free;
        }
        // Triangle is beyond, but created during recent insertion
        else if ( NewValidAndUnchecked == triStatus )
        {
            triStatus = ValidAndUnchecked; // Set it to normal triangle
        }
    }

    return;
}

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
{
    CudaAssert( ( insIdx >= 0 ) && "Invalid insertion index!" );

    // Check if NO exact check needed

    int exactTriCount = beneathData._flagArr[ ExactTriCount ];

    if ( 0 == exactTriCount )
    {
        return; 
    }

    // Check if few OR all triangles need exact check

    const PredicateInfo curThreadPredInfo   = getCurThreadPredInfo( predInfo );
    const bool exactCheckAll                = ( exactTriCount >= ExactTriangleMax ); 

    if ( exactCheckAll )
    {
        exactTriCount = activeTriPosArr._num; 
    }

    // Iterate triangles
    for ( int idx = getCurThreadIdx(); idx < exactTriCount; idx += getThreadNum() )
    {
        ////
        // Check if any insertion for triangle
        ////

        if ( exactCheckAll && ( insIdx >= activeTriInsNumArr._arr[ idx ] ) )
        {
            continue;
        }

        ////
        // Check if triangle needs exact check
        ////

        const TriPosition triPos        = exactCheckAll ? activeTriPosArr._arr[ idx ] : beneathData._exactTriPosArr[ idx ];
        const TriPositionEx triPosEx    = triPosToEx( triPos );
        TriangleStatus& triStatus       = starData.triStatusAt( triPosEx );

        // Ignore triangle not needing exact check
        if ( !triNeedsExactCheck( triStatus ) )
        {
            continue;
        }

        ////
        // Read insertion point
        ////

        const int star      = starData.triStarAt( triPosEx );
        const int insBeg    = insertData._starVertMap[ star ];
        const int insEnd    = ( star < ( starData._starNum - 1 ) ) ? insertData._starVertMap[ star + 1 ] : insertData._vertNum;
        const int insLoc    = insBeg + insIdx;

        CudaAssert( insLoc < insEnd );

        ////
        // Check if triangle beneath point
        ////

        const int insVert   = insertData._vertArr[ insLoc ];
        const Triangle tri  = starData.triangleAt( triPosEx );
        const Orient ord    = orientation4SoS( curThreadPredInfo, pointData, tri._v[0], tri._v[1], tri._v[2], star, insVert );

        // Triangle beneath insertion point
        if ( OrientNeg == ord )
        {
            beneathData._beneathTriPosArr[ star ]   = exToTriPos( triPosEx ); // Store beneath triangle position
            triStatus                               = Free;
        }
        else
        {
            triStatus = ( DoExactOnValid == triStatus ) ? Valid : ValidAndUnchecked; // Set back to old triStatus
        }
    }

    return;
}

////
// The containment proof is star plus 4 points from link of star that encloses input point.
// Returns true if exact check is needed. 
////

template< bool doExact >
__device__ bool
findStarContainmentProof
(
PredicateInfo   curPredInfo,
KerPointData    pointData,
KerStarData     starData,
int             star,       // Star that encloses input point
int             inVert,     // Input point that lies inside star
int*            insStarArr,
int*            insVertArr,
int             insBeg
)
{
    const StarInfo starInfo = starData.getStarInfo( star );

    ////
    // Pick one triangle as facet intersected by plane
    ////

    int locTriIdx = 0;
    
    for ( ; locTriIdx < starInfo._locTriNum; ++locTriIdx )
    {
        const TriPositionEx triPosEx    = starInfo.locToTriPosEx( locTriIdx );
        const TriangleStatus status     = starData.triStatusAt( triPosEx );

        // Ignore free triangles
        if ( Free != status ) break;
    }

    // Pick this valid triangle!
    const TriPositionEx triPosEx    = starInfo.locToTriPosEx( locTriIdx );
    const Triangle& firstTri        = starData.triangleAt( triPosEx );
    const int exVert                = firstTri._v[ 0 ]; // First proof point

    CudaAssert( ( locTriIdx < starInfo._locTriNum ) && "No valid triangle found!" );

    ////
    // Iterate through triangles to find another triangle
    // intersected by plane of (star, inVert, exVert)
    ////

    for ( ; locTriIdx < starInfo._locTriNum; ++locTriIdx )
    {
        // Ignore free triangles
        const TriPositionEx triPosEx    = starInfo.locToTriPosEx( locTriIdx );
        const TriangleStatus status     = starData.triStatusAt( triPosEx );
        
        if ( Free == status ) continue;

        // Ignore triangle if it has exVert

        const Triangle tri = starData.triangleAt( triPosEx );
        
        if ( tri.hasVertex( exVert ) ) continue;

        Orient ord[3];
        int vi = 0; 

        // Iterate through vertices in order
        for ( ; vi < 3; ++vi )
        {
            const int planeVert = tri._v[ vi ];
            const int testVert  = tri._v[ ( vi + 1 ) % 3 ];

            // Get order of testVert against the plane formed by (inVert, starVert, exVert, planeVert)
            Orient order = orientation4Fast( curPredInfo, pointData, star, inVert, exVert, planeVert, testVert );

            if ( OrientZero == order ) 
            {
                if ( doExact ) 
                    order = orientation4SoS( curPredInfo, pointData, star, inVert, exVert, planeVert, testVert );
                else
                    return true; 
            }

            ord[ vi ] = order;

            // Check if orders match, they do if plane intersects facet
            if ( ( vi > 0 ) && ( ord[ vi - 1 ] != ord[ vi ] ) ) break; 
        }

        // All the orders match, we got our proof
        if ( vi >= 3 ) break;
    }

    CudaAssert( ( locTriIdx < starInfo._locTriNum ) && "Could not find proof in star!" );

    ////
    // Write proof vert insertions
    ////

    const TriPositionEx proofTriPosEx   = starInfo.locToTriPosEx( locTriIdx );
    const Triangle proofTri             = starData.triangleAt( proofTriPosEx );

    // First proof point
    insStarArr[ insBeg ] = ( inVert < exVert ) ? inVert : exVert;
    insVertArr[ insBeg ] = ( inVert < exVert ) ? exVert : inVert;

    // Next 3 proof points: write at i+drownedNum, i+2-drownedNum, ...
    for ( int vi = 0; vi < 3; ++vi ) 
    {
        const int triVert   = proofTri._v[ vi ];
        const int insIdx    = insBeg + vi + 1;
        
        insStarArr[ insIdx ] = ( inVert < triVert ) ? inVert  : triVert;
        insVertArr[ insIdx ] = ( inVert < triVert ) ? triVert : inVert;
    }
    
    return false;
}

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
{
    const PredicateInfo curThreadPredInfo = getCurThreadPredInfo( predInfo );

    // Iterate drowned items
    for ( int idx = getCurThreadIdx(); idx < drownedStarArr._num; idx += getThreadNum() )
    {
        const int star  = drownedStarArr._arr[ idx ];  // Killer
        const int vert  = drownedVertArr[ idx ];       // Killed

        ////
        // Go ahead and write the destination star of proof insertions
        // (so no need to write this in exact check)
        ////

        int toIdx = idx * ProofPointsPerStar;

        // Find proof insertions using fast check
        const bool needExact = findStarContainmentProof< false >(
            curThreadPredInfo,
            pointData,
            starData,
            star, vert,
            proofStarArr, proofVertArr,
            toIdx );

        if ( needExact )
        {
            // These will be picked up by exact check
            drownedStarArr._arr[ idx ] = flipToNeg( star );
        }
    }

    return;
}

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
{
    const PredicateInfo curThreadPredInfo = getCurThreadPredInfo( predInfo );

    // Iterate drowned items
    for ( int idx = getCurThreadIdx(); idx < drownedStarArr._num; idx += getThreadNum() )
    {
        // Ignore items whose proof is already found by fast check
        
        const int negStar = drownedStarArr._arr[ idx ];

        if ( negStar >= 0 ) continue;

        const int star  = flipToPos( negStar );
        const int vert  = drownedVertArr[ idx ];
        const int toIdx = ProofPointsPerStar * idx;

        // Write proof vertices

        findStarContainmentProof< true >(
            curThreadPredInfo,
            pointData,
            starData,
            star, vert,
            proofStarArr, proofVertArr,
            toIdx );
    }

    return;
}

__global__ void kerMarkLowerHullTetra
(
PredicateInfo   predInfo,
KerPointData    pointData,
KerStarData     starData,
KerIntArray     tetraTriMap
)
{
    const int tetraNum = tetraTriMap._num;

    // Iterate both lower- and upper-hull tetra
    for ( int tetIdx = getCurThreadIdx(); tetIdx < tetraNum; tetIdx += getThreadNum() )
    {
        // Owner triangle of tetrahedron
        const int triIdx                = tetraTriMap._arr[ tetIdx ];
        const TriPositionEx triPosEx    = starData.globToTriPosEx( triIdx );
        const Triangle tri              = starData.triangleAt( triPosEx );

        ////
        // Orientation of tetra
        // Note: Non-SoS check is enough since flat and -ve tetra are removed
        ////

        const int curStar   = starData.triStarAt( triPosEx );
        const Point3* ptArr = pointData._pointArr;
        const Point3* p[]   = { &( ptArr[ tri._v[0] ] ), &( ptArr[ tri._v[1] ] ), &( ptArr[ tri._v[2] ] ), &( ptArr[ curStar ] ) };
        Orient ord          = shewchukOrient3D( predInfo._consts, p[0]->_p, p[1]->_p, p[2]->_p, p[3]->_p );
        ord                 = flipOrient( ord );

        // Invalidate upper-hull tetra
        if ( OrientPos != ord )
        {
            tetraTriMap._arr[ tetIdx ] = -1;
        }
    }

    return;
}

__global__ void kerMakeCloneFacets
(
KerStarData     starData,
KerIntArray     tetraTriMap,
int*            triTetraMap,
int*            facetStarArr,
int*            facetTriArr
)
{
    const int tetraNum = tetraTriMap._num;

    // Iterate tetra
    for ( int tetIdx = getCurThreadIdx(); tetIdx < tetraNum; tetIdx += getThreadNum() )
    {
        // Owner triangle of tetrahedron
        const int triIdx                = tetraTriMap._arr[ tetIdx ];
        const TriPositionEx triPosEx    = starData.globToTriPosEx( triIdx );
        const Triangle& tri             = starData.triangleAt( triPosEx );

        triTetraMap[ triIdx ]   = tetIdx;       // Map owner triangle to its tetra
        facetStarArr[ tetIdx ]  = tri._v[ 0 ];  // Set facet info
        facetTriArr[ tetIdx ]   = triIdx;       // ...
    }

    return;
}

////////////////////////////////////////////////////////////////////////////////
