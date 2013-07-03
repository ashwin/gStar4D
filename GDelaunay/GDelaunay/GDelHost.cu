/*
Author: Ashwin Nanjappa and Cao Thanh Tung
Filename: GDelHost.cu

Copyright (c) 2013, School of Computing, National University of Singapore. 
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the National University of University nor the names of its contributors
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
//                               Star Host Code
////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////// Headers //

// Project
#include "Config.h"
#include "Pba.h"
#include "Geometry.h"
#include "PerfTimer.h"
#include "GDelData.h"
#include "GDelKernels.h"
#include "GDelCommon.h"

///////////////////////////////////////////////////////////////////// Globals //

////
// Note: Do NOT set these values!
// They are set by init function.
////

const float DataExpansionFactor = 1.3f;

int* DGrid              = NULL;
int GridWidth           = 0;
int ThreadsPerBlock     = -1;
int BlocksPerGrid       = -1;
int ThreadNum           = -1;
int PredThreadsPerBlock = -1;
int PredBlocksPerGrid   = -1;
int PredThreadNum       = -1;
int InsertNum           = -1;
int WorksetSizeMax      = -1;
int InsertPointMax      = -1;
int LoopNum             = -1;
int FacetMax            = -1;

bool DoSorting  = false;
bool DoCheck    = false;
bool LogVerbose = false;
bool LogStats   = false;
bool LogTiming  = false;
bool LogMemory  = false;

// Containers
ActiveData      DActiveData;
BeneathData     DBeneathData;
HistoryData     DHistoryData;
InsertionData   DInsertData;
MissingData     DMissingData;
PointData       DPointData;
PredicateInfo   DPredicateInfo;
StarData        DStarData;
TetraData       DTetraData;

// Device vectors
IntDVec* DTriBufVec; // Used as scratch vector by many functions

// Host vectors
TetraHVec HTetraVec;  // Used to store final tetra

////////////////////////////////////////////////////////////////// Probe data //

HostTimer loopTimer;
HostTimer insertTimer;
HostTimer expandTimer;
HostTimer sortTimer;
HostTimer initTimer;
HostTimer drownTimer;
HostTimer getInsertTimer;

double expandTime       = 0;
double insertTime       = 0;
double sortTime         = 0;
double drownTime        = 0;
double getInsertTime    = 0;

int LogStarInsNum  = 0;
int LogSplayInsNum = 0;
int LogLoopNum     = 0;

////////////////////////////////////////////////////////////////////////////////
//                                Stars Init
////////////////////////////////////////////////////////////////////////////////

void initPredicate()
{
    DPredicateInfo.init();

    // Predicate constants
    DPredicateInfo._consts = cuNew< RealType >( DPredicateBoundNum );

    // Predicate arrays
    DPredicateInfo._data = cuNew< RealType >( PredicateTotalSize * PredThreadNum );

    // Set predicate constants
    kerInitPredicate<<< 1, 1 >>>( DPredicateInfo._consts );
    CudaCheckError();

    return;
}

void starsInit
(
Point3HVec&     pointHVec,
Point3HVec&     scaledHVec,
const Config&   config,
Point3DVec**    outPointDVec
)
{
    GridWidth   = config._gridSize;
    FacetMax    = config._facetMax;
    LogVerbose  = config._logVerbose;
    LogStats    = config._logStats;
    LogTiming   = config._logTiming;
    DoSorting   = config._doSorting;
    DoCheck     = config._doCheck;

    ThreadsPerBlock = MAX_THREADS_PER_BLOCK;
    BlocksPerGrid   = 512;
    ThreadNum       = ThreadsPerBlock * BlocksPerGrid;

    PredThreadsPerBlock = ( config._predThreadNum < 0 ) ? 32 : config._predThreadNum;
    PredBlocksPerGrid   = ( config._predBlockNum < 0 )  ? 32 : config._predBlockNum;
    PredThreadNum       = PredThreadsPerBlock * PredBlocksPerGrid;

    // Sanity check
    assert( ThreadsPerBlock <= MAX_THREADS_PER_BLOCK );
    assert( PredThreadsPerBlock <= MAX_PRED_THREADS_PER_BLOCK );
    assert( PredBlocksPerGrid >= MIN_BLOCKS_PER_MP );

    initPredicate();

    ////
    // Set kernels to prefer L1 cache
    ////

    CudaSafeCall( cudaFuncSetCacheConfig( kerAppendValueToKey,          cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerCheckStarConsistency,      cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerComputeTriangleCount,      cudaFuncCachePreferL1 ) ); 
    CudaSafeCall( cudaFuncSetCacheConfig( kerConvertMapToCount,         cudaFuncCachePreferL1 ) ); 
    CudaSafeCall( cudaFuncSetCacheConfig( kerCopyInsertionToNewHistory, cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerCopyOldToNewHistory,       cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerCopyPumpedUsingAtomic,     cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerCopyWorksets,              cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerCountPerStarInsertions,    cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerCountPointsOfStar,         cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerGetActiveTriPos,           cudaFuncCachePreferL1 ) ); 
    CudaSafeCall( cudaFuncSetCacheConfig( kerGetActiveTriInsCount,      cudaFuncCachePreferL1 ) ); 
    CudaSafeCall( cudaFuncSetCacheConfig( kerGetCloneTriInfo,           cudaFuncCachePreferL1 ) ); 
    CudaSafeCall( cudaFuncSetCacheConfig( kerGetProofExact,             cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerGetProofFast,              cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerGatherPumpedInsertions,    cudaFuncCachePreferL1 ) ); 
    CudaSafeCall( cudaFuncSetCacheConfig( kerGetPerTriangleInsertions,  cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerGetPerTriangleCount,       cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerGrabTetrasFromStars,       cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerMakeInitialConeExact,      cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerMakeInitialConeFast,       cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerMakeCloneFacets,           cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerMakeMissingData,           cudaFuncCachePreferL1 ) ); 
    CudaSafeCall( cudaFuncSetCacheConfig( kerMakeOldToNewTriMap,        cudaFuncCachePreferL1 ) ); 
    CudaSafeCall( cudaFuncSetCacheConfig( kerMarkBeneathTrianglesExact, cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerMarkBeneathTrianglesFast,  cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerMarkDuplicateDrownedPairs, cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerMarkDuplicates,            cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerMarkIfInsertionInHistory,  cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerMarkLowerHullTetra,        cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerMarkReversePairs,          cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerMarkSubmergedInsertions,   cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerNoteOwnerTriangles,        cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerOrderDrownedPairs,         cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerReadPairsFromGrid,         cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerRemoveValueFromKey,        cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerRestoreDrownedPairs,       cudaFuncCachePreferL1 ) );
    CudaSafeCall( cudaFuncSetCacheConfig( kerStitchPointToHole,         cudaFuncCachePreferL1 ) ); 
    
    // Move points to device and sort them
    DPointData.init( pointHVec, scaledHVec );

    ////
    // Other stuff
    ////

    const int pointNum = ( int ) DPointData._pointVec->size();
    getPool().init( pointNum * sizeof( int ) ); 

    DActiveData.init();
    DBeneathData.init( pointNum );
    DHistoryData.init();
    DInsertData.init();
    DMissingData.init();
    DStarData.init( pointNum );
    DTetraData.init();

    DTriBufVec = new IntDVec();
    
    // Return device pointer to points
    *outPointDVec = scaledHVec.empty() ? DPointData._pointVec : DPointData._scaledVec;

    return;
}

void starsDeinit()
{
    DActiveData.deInit();
    DBeneathData.deInit();
    DHistoryData.deInit();
    DInsertData.deInit();
    DMissingData.deInit();
    DPointData.deinit();
    DPredicateInfo.deInit();
    DStarData.deInit();
    DTetraData.deInit();

    safeDeleteDevConPtr( &DTriBufVec );
    
    getPool().deInit();

    return;
}

////////////////////////////////////////////////////////////////////////////////
//                         makeStarsFromGrid()
////////////////////////////////////////////////////////////////////////////////

void _collectMissingData()
{
    const int pointNum = ( int ) DPointData._pointVec->size();

    // Prepare for missing data
    DMissingData._memberVec->resize( pointNum );
    DMissingData._leaderVec->resize( pointNum );

    // Create member list (list of all points)
    thrust::sequence( DMissingData._memberVec->begin(), DMissingData._memberVec->end() );

    // Read grid to find leader of each point
    kerMakeMissingData<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DGrid,
        GridWidth,
        DPointData.toKernel(),
        DMissingData.toKernel() );
    CudaCheckError();

    // Remove point-pairs which are winners (voted for themselves)
    compactBothIfEqual( *DMissingData._memberVec, *DMissingData._leaderVec );

    if ( LogVerbose )
    {
        cout << "Missing points: " << DMissingData._memberVec->size() << endl;
    }

    return;
}

void _readGridPairs()
{
    ////
    // Count pairs
    ////

    const int BlocksPerGrid     = GridWidth + 2; 
    const int ThreadsPerBlock   = GridWidth; 
    const int ThreadNum         = BlocksPerGrid * ThreadsPerBlock; 

    // Use this array to gather count of pairs-per-thread
    DInsertData._vertVec->resize( ThreadNum ); 

    // Get per-thread pair count
    kerReadPairsFromGrid<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DGrid,
        GridWidth,
        DInsertData.toKernel(),
        CountPerThreadPairs );
    CudaCheckError();

    // Convert count to per-thread map
    const int worksetNum = makeMapAndSum( *DInsertData._vertVec, *DInsertData._starVertMap );

    ////
    // Grab pairs
    ////

    // Prepare workset array
    DInsertData._vertVec->resize( worksetNum );
    DInsertData._vertStarVec->resize( worksetNum );

    if ( LogVerbose )
    {
        cout << "Workset pairs: " << DInsertData._vertVec->size() << " (before sort and unique)" << endl;
    }

    // Read pairs from grid
    kerReadPairsFromGrid<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DGrid,
        GridWidth,
        DInsertData.toKernel(),
        GrabPerThreadPairs );
    CudaCheckError();

    // Output Voronoi diagram is no longer useful, so free it
    CudaSafeCall( cudaFree( DGrid ) );

    ////
    // Sort workset pairs and remove duplicates
    ////

    // 80ms
    sortAndUniqueUsingKeyValAppend( *DInsertData._vertStarVec, *DInsertData._vertVec, DPointData._bitsPerIndex );

    if ( LogVerbose )
    {
        cout << "Workset pairs: " << DInsertData._vertVec->size() << " (after sort and unique)" << endl;
    }

    return;
}

void _pumpWorksets()
{
    ////
    // Create array with capacity to hold:
    // <-- curWorkset --><-- curWorkset (flipped) --><-- missing --><-- missing (flipped) -->
    ////

    const int compactWorksetNum = DInsertData._vertVec->size();
    const int missingNum        = DMissingData._memberVec->size();
    const int pairNum           = compactWorksetNum + missingNum;
    const int tmpWorksetNum     = 2 * pairNum;

    IntDVec wsetVertVec( tmpWorksetNum );
    IntDVec wsetStarVec( tmpWorksetNum );

    ////
    // Make flipped copies of worksets and missing data
    ////

    // Copy workset pairs
    thrust::copy( DInsertData._vertStarVec->begin(), DInsertData._vertStarVec->end(), wsetStarVec.begin() );
    thrust::copy( DInsertData._vertVec->begin(),     DInsertData._vertVec->end(),     wsetVertVec.begin() );

    // Copy missing pairs
    thrust::copy( DMissingData._memberVec->begin(), DMissingData._memberVec->end(), wsetStarVec.begin() + compactWorksetNum );
    thrust::copy( DMissingData._leaderVec->begin(), DMissingData._leaderVec->end(), wsetVertVec.begin() + compactWorksetNum );

    // Reciprocate the pairs
    thrust::copy( wsetVertVec.begin(), wsetVertVec.begin() + pairNum, wsetStarVec.begin() + pairNum );
    thrust::copy( wsetStarVec.begin(), wsetStarVec.begin() + pairNum, wsetVertVec.begin() + pairNum );

    // Missing data is not needed anymore
    DMissingData.deInit();

    if ( LogVerbose )
    {
        cout << "Workset pairs: " << tmpWorksetNum << " (after missing and reciprocated)" << endl;
    }

    ////
    // Sort and make map of worksets
    ////

    // 20 ms
    thrust::sort_by_key( wsetStarVec.begin(), wsetStarVec.end(), wsetVertVec.begin() );

    const int pointNum = DPointData._pointVec->size();

    IntDVec wsetCountVec;
    IntDVec wsetMap; // Used later below

    makeAllStarMap( wsetStarVec, wsetMap, pointNum );
    convertMapToCountVec( wsetMap, wsetCountVec, wsetStarVec.size() );

    ////
    // Make pump map *only* for stars that need pumping
    ////

    IntDVec actStarVec( pointNum );
    thrust::sequence( actStarVec.begin(), actStarVec.end() );

    IntDVec pumpMap( pointNum );
    thrust::transform( wsetCountVec.begin(), wsetCountVec.end(), pumpMap.begin(), GetPumpedWorksetSize() );

    compactBothIfZero( pumpMap, actStarVec );

    const int pumpNum = makeInPlaceMapAndSum( pumpMap );

    ////
    // Get unique reciprocated pumped insertions and make map
    ////

    // Create with space for pumped + reciprocated pumped
    IntDVec pumpStarVec( 2 * pumpNum ); 
    IntDVec pumpVertVec( 2 * pumpNum );

    // Resize it back to required size
    pumpStarVec.resize( pumpNum );
    pumpVertVec.resize( pumpNum );

    kerGatherPumpedInsertions<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( wsetMap ),
        toKernelArray( wsetStarVec ),
        toKernelPtr( wsetVertVec ),
        toKernelArray( actStarVec ),
        toKernelPtr( pumpMap ),
        toKernelArray( pumpStarVec ),
        toKernelPtr( pumpVertVec ) );
    CudaCheckError();

    // Write reciprocated pump insertions

    pumpStarVec.resize( 2 * pumpNum );
    pumpVertVec.resize( 2 * pumpNum );

    thrust::copy( pumpStarVec.begin(), pumpStarVec.begin() + pumpNum, pumpVertVec.begin() + pumpNum );
    thrust::copy( pumpVertVec.begin(), pumpVertVec.begin() + pumpNum, pumpStarVec.begin() + pumpNum );

    // 3 ms
    sortAndUniqueUsingKeyValAppend( pumpStarVec, pumpVertVec, DPointData._bitsPerIndex );

    const int compRecPumpNum = pumpStarVec.size();

    // Create pump count per star
    IntDVec pumpCountVec( pointNum );
    makeAllStarCountVec( pumpStarVec, pumpCountVec, pointNum );

    ////
    // Get insertions per star
    ////

    DStarData._insCountVec->resize( pointNum );

    thrust::transform(
        wsetCountVec.begin(), wsetCountVec.end(),   // From
        pumpCountVec.begin(),                       // From
        DStarData._insCountVec->begin(),            // To
        thrust::plus< int >() );

    WorksetSizeMax = *( thrust::max_element( DStarData._insCountVec->begin(), DStarData._insCountVec->end() ) );

    makeMapAndSum( pumpCountVec, pumpMap );

    ////
    // Make final workset map
    // finalMap = wsetMap + pumpMap
    ////

    DInsertData._starVertMap->resize( pointNum );

    thrust::transform(
        wsetMap.begin(), wsetMap.end(),     // From-1
        pumpMap.begin(),                    // From-2
        DInsertData._starVertMap->begin(),  // To = From-1 + From-2
        thrust::plus< int >() );

    const int finalWorksetNum = tmpWorksetNum + compRecPumpNum;

    ////
    // Copy worksets and pumps to final array
    ////

    DInsertData._vertStarVec->resize( finalWorksetNum );
    DInsertData._vertVec->resize( finalWorksetNum );

    kerCopyWorksets<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DInsertData.toKernel(),
        toKernelArray( wsetStarVec ),
        toKernelPtr( wsetVertVec ),
        toKernelPtr( wsetMap ) );
    CudaCheckError();

    // Destination index for pumped insertions
    thrust::transform(
        wsetCountVec.begin(), wsetCountVec.end(),   // From
        DInsertData._starVertMap->begin(),          // From
        wsetCountVec.begin(),                       // To
        thrust::plus< int >() );

    kerCopyPumpedUsingAtomic<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DInsertData.toKernel(),
        toKernelArray( pumpStarVec ),
        toKernelPtr( pumpVertVec ),
        toKernelPtr( wsetCountVec ) );
    CudaCheckError();

    if ( LogVerbose )
    {
        cout << "Largest working set: " << WorksetSizeMax << endl;
        cout << "Workset pairs: " << DInsertData._vertVec->size() << endl;
        cout << "Average workset per star: " << DInsertData._vertVec->size() / DInsertData._starVertMap->size() << endl;
    }

    if ( LogStats )
        LogStarInsNum = DInsertData._vertVec->size();

    return;
}

// Initialize history using initial insertions
void _initHistory()
{
    DHistoryData._vertVec[ 0 ]->copyFrom( *DInsertData._vertVec );
    DHistoryData._vertStarVec[ 0 ]->copyFrom( *DInsertData._vertStarVec );
    DHistoryData._starVertMap[ 0 ]->copyFrom( *DInsertData._starVertMap );
    DHistoryData._starVertMap[ 1 ]->resize( DInsertData._starVertMap->size(), 0 );
    
    return;
}

// Create 4-simplex for every star
void _makeInitialCones()
{
    if ( LogVerbose )
    {
        cout << endl << __FUNCTION__ << endl;
    }

    ////
    // Prepare star arrays
    ////

    DStarData._starNum  = DInsertData._starVertMap->size();
    const int triNum    = get2SphereTriangleNum( DStarData._starNum, DInsertData._vertVec->size() );

    DStarData._triData.resize( triNum, 0, Free ); // Allocate only array[0] in the beginning

    // Buffer to reuse for triangle related arrays
    const int expTriNum = ( int ) ( triNum * DataExpansionFactor );
    DTriBufVec->resize( expTriNum );

    if ( LogStats )
    {
        cout << "Triangles allocated: " << triNum << endl;
    }

    ////
    // Create initial 4-simplex for each star
    ////

    kerMakeInitialConeFast<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DPredicateInfo,
        DPointData.toKernel(),
        DStarData.toKernel(),
        DBeneathData.toKernel(),
        DInsertData.toKernel() );
    CudaCheckError();

    kerMakeInitialConeExact<<< PredBlocksPerGrid, PredThreadsPerBlock >>>(
        DPredicateInfo,
        DPointData.toKernel(),
        DStarData.toKernel(),
        DBeneathData.toKernel(),
        DInsertData.toKernel() ); 
    CudaCheckError();

    return;
}

void __addOnePointToStar
(
TriPositionDVec&    activeTriPosVec,
ShortDVec&          activeTriInsNumVec,
int                 insIdx
)
{
    kerMarkBeneathTrianglesFast<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DPredicateInfo,
        DPointData.toKernel(),
        DStarData.toKernel(),
        DBeneathData.toKernel(),
        DInsertData.toKernel(),
        toKernelArray( activeTriPosVec ), 
        toKernelArray( activeTriInsNumVec ),
        insIdx );
    CudaCheckError();

    if ( LogStats )
    {
        if ( (*DBeneathData._flagVec)[ ExactTriCount ] > 0 ) 
        {
            cout << "Exact check triangles : " << (*DBeneathData._flagVec)[ ExactTriCount ] << endl; 
        }
    }

    kerMarkBeneathTrianglesExact<<< PredBlocksPerGrid, PredThreadsPerBlock >>>(
        DPredicateInfo,
        DPointData.toKernel(),
        DStarData.toKernel(),
        DBeneathData.toKernel(),
        DInsertData.toKernel(),
        toKernelArray( activeTriPosVec ), 
        toKernelArray( activeTriInsNumVec ),
        insIdx );
    CudaCheckError();

    kerStitchPointToHole<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        DBeneathData.toKernel(),
        DInsertData.toKernel(),
        toKernelArray( DActiveData._starVec ),
        insIdx );
    CudaCheckError(); 

    return;
}

void updateActiveTriData
(
TriPositionDVec&    activeTriPosVec,
ShortDVec&          activeTriInsNumVec
)
{
    kerGetActiveTriPos<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        toKernelArray( DActiveData._starVec ), 
        toKernelArray( DActiveData._starTriMap ), 
        toKernelArray( activeTriPosVec ),
        toKernelArray( activeTriInsNumVec ) );
    CudaCheckError();

    kerGetActiveTriInsCount<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        toKernelArray( activeTriPosVec ), 
        toKernelArray( activeTriInsNumVec ) );
    CudaCheckError();

    return;
}

void _makeStars()
{
    if ( LogVerbose )
    {
        cout << endl << __FUNCTION__ << endl;
    }
    
    ////
    // Prepare to work only on active triangles
    ////

    DActiveData._starVec->resize( DStarData._starNum );  // Assume all stars as active in beginning
    DActiveData._starTriMap->resize( DStarData._starNum );

    thrust::sequence( DActiveData._starVec->begin(), DActiveData._starVec->end() );

    int activeTriNum = DStarData._triData.totalSize(); 

    ////
    // Insert workset points into stars
    ////

    bool isActiveBoundTight             = true; // Cannot be false! Will not work with false!
    TriPositionDVec& activeTriPosVec    = *DTriBufVec;
    ShortDVec activeTriInsNumVec( activeTriNum );

    DBeneathData._flagVec->fill( 0 );
    DBeneathData._beneathTriPosVec->fill( -1 );

    for ( int insIdx = 4; insIdx < WorksetSizeMax; ++insIdx )
    {
        if ( isActiveBoundTight ) 
        {
            kerGetActiveTriCount<<< BlocksPerGrid, ThreadsPerBlock >>>(
                DStarData.toKernel(),
                toKernelArray( DActiveData._starVec ),
                toKernelArray( DActiveData._starTriMap ),
                insIdx,
                isActiveBoundTight ); 
            CudaCheckError();

            compactBothIfZero( *DActiveData._starTriMap, *DActiveData._starVec );

            activeTriNum = makeInPlaceMapAndSum( *DActiveData._starTriMap );

            // Turn OFF careful mode when too few active triangles
            if ( activeTriNum < ThreadNum )
            {
                isActiveBoundTight = false; 

                // Careful mode has been turned OFF
                // Get loose bound of active triangles one last time
                kerGetActiveTriCount<<< BlocksPerGrid, ThreadsPerBlock >>>(
                    DStarData.toKernel(),
                    toKernelArray( DActiveData._starVec ),
                    toKernelArray( DActiveData._starTriMap ),
                    insIdx,
                    isActiveBoundTight ); 
                CudaCheckError();

                activeTriNum = makeInPlaceMapAndSum( *DActiveData._starTriMap );
            }

            if ( LogStats ) 
            {
                cout << "activeStar = " << DActiveData._starVec->size() << ", activeTri = " << activeTriNum << endl; 
            }

            activeTriPosVec.resize( activeTriNum ); 
            activeTriInsNumVec.resize( activeTriNum ); 

            updateActiveTriData( activeTriPosVec, activeTriInsNumVec );
        }

        __addOnePointToStar( activeTriPosVec, activeTriInsNumVec, insIdx );
    }

    kerCountPointsOfStar<<< BlocksPerGrid, ThreadsPerBlock >>>( DStarData.toKernel() ); 
    CudaCheckError(); 

    return;
}

// Initialize link vertices of each star from tetra.
void makeStarsFromGrid( int* grid )
{
    DGrid = grid;

    if ( LogTiming ) 
    {
        initTimer.start(); 
    }

    _collectMissingData();

    if ( LogTiming ) 
    {
        initTimer.stop(); 
        initTimer.print( "_collectMissingData" ); 
        initTimer.start(); 
    }

    _readGridPairs();
    
    if ( LogTiming ) 
    {
        initTimer.stop(); 
        initTimer.print( "_readGridPairs" ); 
        initTimer.start(); 
    }

    _pumpWorksets();

    if ( LogTiming ) 
    {
        initTimer.stop(); 
        initTimer.print( "_pumpWorksets" ); 
        initTimer.start(); 
    }

    _initHistory();

    if ( LogTiming ) 
    {
        initTimer.stop(); 
        initTimer.print( "_initHistory" ); 
        initTimer.start(); 
    }

    _makeInitialCones();

    if ( LogTiming ) 
    {
        initTimer.stop(); 
        initTimer.print( "_makeInitialCones" ); 
        initTimer.start(); 
    }

    _makeStars();

    if ( LogTiming ) 
    {
        initTimer.stop(); 
        initTimer.print( "_makeStars" );
        cout << endl;
    }

    return;
}

////////////////////////////////////////////////////////////////////////////////
//                            processFacets()
////////////////////////////////////////////////////////////////////////////////

// Drowned are insertions not present in link of star
void __getDrownedInsertions()
{
    // Insertions found inside cone are already marked
    // Now mark insertions which are not in link of star
    kerMarkSubmergedInsertions<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        DInsertData.toKernel() );
    CudaCheckError();

    // Remove all insertions except drowned
    compactBothIfNegative( *DInsertData._vertStarVec, *DInsertData._vertVec );

    return;
}

void __removeMutualDrownedInsertions()
{
    // Order pairs so that (star < vert) and make key as key-val
    kerOrderDrownedPairs<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( DInsertData._vertStarVec ),
        toKernelPtr( DInsertData._vertVec ),
        DPointData._bitsPerIndex );
    CudaCheckError();

    // Sort by key that is actually key-val
    thrust::sort_by_key(
        DInsertData._vertStarVec->begin(), DInsertData._vertStarVec->end(), // Key
        DInsertData._vertVec->begin() );                                    // Val

    // If dup found, mark both original *and* duplicate
    kerMarkDuplicateDrownedPairs<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( DInsertData._vertStarVec ),
        toKernelPtr( DInsertData._vertVec ) );
    CudaCheckError();

    // Remove duplicates (both original and its match)
    compactBothIfNegative( *DInsertData._vertVec, *DInsertData._vertStarVec );

    // Restore back pairs to original order
    kerRestoreDrownedPairs<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( DInsertData._vertStarVec ),
        toKernelPtr( DInsertData._vertVec ),
        DPointData._bitsPerIndex ); 
    CudaCheckError();

    const int drownedNum = DInsertData._vertVec->size();

    if ( drownedNum > ThreadNum )
    {
        thrust::sort_by_key( DInsertData._vertStarVec->begin(), DInsertData._vertStarVec->end(), DInsertData._vertVec->begin() );
    }   

    if ( LogStats )
    {
        cout << "Drowned: " << drownedNum << endl;
    }

    return;
}

void _getDrownedInsertions()
{
    if ( LogTiming )
    {
        drownTimer.start();
    }

    __getDrownedInsertions();
    __removeMutualDrownedInsertions();

    if ( LogTiming )
    {
        drownTimer.stop();
        drownTime += drownTimer.value();
    }

    return;
}

int __gatherInsertions()
{
    IntDVec proofStarVec;
    IntDVec proofVertVec;

    const int drownedNum = DInsertData._vertVec->size();

    if ( drownedNum > 0 )
    {
        /////
        // Gather proof insertions
        ////

        const int proofNum = ProofPointsPerStar * drownedNum;

        proofStarVec.resize( proofNum );
        proofVertVec.resize( proofNum );

        kerGetProofFast<<< BlocksPerGrid, ThreadsPerBlock >>>(
            DPredicateInfo,
            DPointData.toKernel(),
            DStarData.toKernel(),
            toKernelArray( DInsertData._vertStarVec ),
            toKernelPtr( DInsertData._vertVec ),
            toKernelPtr( proofStarVec ),
            toKernelPtr( proofVertVec ) );
        CudaCheckError();

        kerGetProofExact<<< PredBlocksPerGrid, PredThreadsPerBlock >>>(
            DPredicateInfo,
            DPointData.toKernel(),
            DStarData.toKernel(),
            toKernelArray( DInsertData._vertStarVec ),
            toKernelPtr( DInsertData._vertVec ),
            toKernelPtr( proofStarVec ),
            toKernelPtr( proofVertVec ) );
        CudaCheckError();

        if ( LogStats )
            cout << "Proof insertions: " << proofNum << endl;
    }

    ////
    // Get insertion count from triangles
    ////

    IntDVec& triInsertMap = *DTriBufVec;
    triInsertMap.resize( DStarData._triData.totalSize() );

    kerGetPerTriangleCount<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        toKernelPtr( triInsertMap ) );
    CudaCheckError();

    const int triInsertNum          = makeInPlaceMapAndSum( triInsertMap );
    const int boundedTriInsertNum   = std::min( triInsertNum, FacetMax );

    const int proofNum  = proofStarVec.size();
    int insertNum       = proofNum + boundedTriInsertNum;

    if ( LogStats )
    {
        cout << "Triangle insertions:         " << triInsertNum << endl;
        cout << "Bounded triangle insertions: " << boundedTriInsertNum << endl;
    }

    if ( 0 == insertNum )
    {
        return insertNum; // No insertions to do
    }

    ////
    // Gather triangle insertions
    ////

    if ( 0 == proofNum )
    {
        ++insertNum; // +1 space to hold insertion of triangle on FacetMax boundary
    }
    else
    {
        // +1 space of proof insertions, it is overwritten later anyway
    }

    if ( DInsertData._vertVec->capacity() < insertNum )
    {
        DInsertData._vertStarVec->resize( insertNum );
        DInsertData._vertVec->resize( insertNum );
    }

    DInsertData._vertStarVec->expand( boundedTriInsertNum );
    DInsertData._vertVec->expand( boundedTriInsertNum );

    kerGetPerTriangleInsertions<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        toKernelArray( triInsertMap ),
        toKernelArray( DInsertData._vertStarVec ),
        toKernelPtr( DInsertData._vertVec ),
        triInsertNum );
    CudaCheckError();

    ////
    // Collect all insertions together
    ////

    DInsertData._vertStarVec->expand( insertNum ); // insertNum = boundedTriInsertNum + proofNum
    DInsertData._vertVec->expand( insertNum );

    // Copy proof insertions
    thrust::copy( proofStarVec.begin(), proofStarVec.end(), DInsertData._vertStarVec->begin() + boundedTriInsertNum );
    thrust::copy( proofVertVec.begin(), proofVertVec.end(), DInsertData._vertVec->begin() + boundedTriInsertNum );

    // Sort and remove duplicates of ordered pairs
    sortAndUniqueUsingKeyValAppend( *DInsertData._vertStarVec, *DInsertData._vertVec, DPointData._bitsPerIndex );

    ////
    // Remove insertions if already in history
    ////

    kerMarkIfInsertionInHistory<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DHistoryData.toKernel(),
        toKernelArray( DInsertData._vertStarVec ),
        toKernelPtr( DInsertData._vertVec ),
        DStarData._starNum );
    CudaCheckError();

    compactBothIfNegative( *DInsertData._vertVec, *DInsertData._vertStarVec );

    const int compInsertNum = DInsertData._vertVec->size();

    ////
    // Reciprocate insertions
    ////

    const int recInsertNum = 2 * compInsertNum;

    // Expand if needed
    if ( DInsertData._vertVec->capacity() < recInsertNum )
    {
        cout << "This should almost never happen! Remove this cout when it happens." << endl;

        IntDVec tmpStarVec( recInsertNum );
        IntDVec tmpVertVec( recInsertNum );

        thrust::copy( DInsertData._vertStarVec->begin(), DInsertData._vertStarVec->end(), tmpStarVec.begin() );
        thrust::copy( DInsertData._vertVec->begin(),     DInsertData._vertVec->end(),     tmpVertVec.begin() );

        DInsertData._vertStarVec->swapAndFree( tmpStarVec );
        DInsertData._vertVec->swapAndFree( tmpVertVec );
    }
    else
    {
        DInsertData._vertStarVec->expand( recInsertNum );
        DInsertData._vertVec->expand( recInsertNum );
    }

    thrust::copy( DInsertData._vertStarVec->begin(), DInsertData._vertStarVec->begin() + compInsertNum, DInsertData._vertVec->begin() + compInsertNum );
    thrust::copy( DInsertData._vertVec->begin(),     DInsertData._vertVec->begin() + compInsertNum,     DInsertData._vertStarVec->begin() + compInsertNum );

    if ( LogStats )
    {
        cout << "Reciprocated insertions: " << recInsertNum << endl;
    }

    return recInsertNum;
}

int _gatherInsertions()
{
    if ( LogTiming )
    {
        getInsertTimer.start();
    }

    const int insNum = __gatherInsertions();

    if ( LogTiming )
    {
        getInsertTimer.stop();
        getInsertTime += getInsertTimer.value();
    }

    return insNum;
}

void __sortInsertions()
{
    // There cannot be any duplicates since we used ordered pairs
    // Only sort is enough
    thrust::sort_by_key(
        DInsertData._vertStarVec->begin(), DInsertData._vertStarVec->end(), // Key
        DInsertData._vertVec->begin() );                                    // Val

    if ( LogVerbose )
    {
        cout << "Unique insertions: " << DInsertData._vertVec->size() << endl;
    }

    makeAllStarMap( *DInsertData._vertStarVec, *DInsertData._starVertMap, DStarData._starNum );
    convertAllStarMapToSimpleMap( *DInsertData._starVertMap, DInsertData._vertStarVec->size() );

    ////
    // Find insertion point count for each star
    ////

    // Go back and update the point count of each star (current count + intended insertion count)
    kerCountPerStarInsertions<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        DInsertData.toKernel() );
    CudaCheckError();

    ////
    // Find largest insertion point count
    ////

    InsertPointMax = *( thrust::max_element( DStarData._insCountVec->begin(), DStarData._insCountVec->end() ) );

    if ( LogVerbose )
    {
        cout << "Largest insertion set: " << InsertPointMax << endl;
    }

    return;
}

void _sortInsertions()
{
    if ( LogTiming )
    {
        sortTimer.start();
    }

    __sortInsertions();

    if ( LogTiming )
    {
        sortTimer.stop();
        sortTime += sortTimer.value();
    }

    return;
}

void __expandStarsForInsertion()
{
    if ( LogStats )
    {
        cout << endl << __FUNCTION__ << endl;
    }

    ////
    // Calculate triangle/segment count for insertion
    ////

    // Prepare triangle count array
    IntDVec dTriNumVec( DStarData._starNum );

    // Estimate triangles needed for insertion
    kerComputeTriangleCount<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        toKernelArray( dTriNumVec ) );
    CudaCheckError();

    ////
    // Expand triangles *only* if needed
    ////

    // Compute new triangles map and sum

    IntDVec newTriMap;

    const int newTriNum = makeMapAndSum( dTriNumVec, newTriMap );

    dTriNumVec.free();

    // Check if triangle array 2 needs to be expanded

    const int curTriNum = ( int ) DStarData._triData.size( 1 );

    if ( curTriNum < newTriNum )
    {
        if ( LogStats )
        {
            cout << "Expanding triangles From: " << curTriNum << " To: " << newTriNum << endl;
        }

        DStarData.expandTriangles( newTriNum, newTriMap );
    }

    return;
}

void _expandStarsForInsertion()
{
    if ( LogTiming )
    {
        expandTimer.start();
    }

    __expandStarsForInsertion();

    if ( LogTiming )
    {
        expandTimer.stop();
        expandTime += expandTimer.value();
    }

    return;
}

// History[0] arrays remain as they are
// Only history[1] arrays are updated here
void __copyInsertionsToHistory()
{
    if ( LogStats )
    {
        cout << __FUNCTION__ << endl;
    }

    const int insHistNum = DInsertData._vertVec->size();
    const int oldHistNum = DHistoryData._vertVec[1]->size();
    const int newHistNum = oldHistNum + insHistNum;

    assert( ( 0 == ( insHistNum % 2 ) ) && "Should be even number since insertions are reciprocated!" );

    // History[1] does not exist
    if ( 0 == oldHistNum )
    {
        DHistoryData._vertVec[1]->copyFrom( *DInsertData._vertVec );
        DHistoryData._vertStarVec[1]->copyFrom( *DInsertData._vertStarVec );
        DHistoryData._starVertMap[1]->copyFrom( *DInsertData._starVertMap );
    }
    else // History[1] already exists
    {
        IntDVec newVertVec( newHistNum );
        IntDVec newStarVec( newHistNum );
        IntDVec newMap( DStarData._starNum );

        // Make destination map
        thrust::transform(
            DInsertData._starVertMap->begin(), DInsertData._starVertMap->end(), // From-1
            DHistoryData._starVertMap[1]->begin(),                              // From-2
            newMap.begin(),                                                     // Res = From-1 + From-2
            thrust::plus< int >() );

        // Copy old history to destination
        kerCopyOldToNewHistory<<< BlocksPerGrid, ThreadsPerBlock >>>(
            DHistoryData.toKernel(),
            toKernelPtr( newVertVec ),
            toKernelPtr( newStarVec ),
            toKernelPtr( newMap ) );
        CudaCheckError();

        // Copy insertions to destination
        kerCopyInsertionToNewHistory<<< BlocksPerGrid, ThreadsPerBlock >>>(
            toKernelArray( DInsertData._vertVec ),
            toKernelPtr( DInsertData._vertStarVec ),
            toKernelArray( DInsertData._starVertMap ),
            toKernelPtr( DHistoryData._starVertMap[1] ),
            oldHistNum,
            toKernelPtr( newVertVec ),
            toKernelPtr( newStarVec ),
            toKernelPtr( newMap ) );
        CudaCheckError();

        DHistoryData._vertVec[1]->swapAndFree( newVertVec );
        DHistoryData._vertStarVec[1]->swapAndFree( newStarVec );
        DHistoryData._starVertMap[1]->swapAndFree( newMap );
    }

    return;
}

void __insertPointsToStars()
{
    if ( LogStats )
        cout << endl << __FUNCTION__ << endl;

    ////
    // Prepare to work only on active triangles
    // "Active" triangles/stars are those that have some insertion
    ////

    // Find active stars

    DActiveData._starVec->resize( DStarData._starNum );

    thrust::sequence( DActiveData._starVec->begin(), DActiveData._starVec->end() );
    compactIfZero( *DActiveData._starVec, *DStarData._insCountVec );

    DActiveData._starTriMap->resize( DActiveData._starVec->size() ); 

    // Find triangle count for each active star
    kerGetActiveTriCount<<< BlocksPerGrid, ThreadsPerBlock >>>( 
        DStarData.toKernel(), 
        toKernelArray( DActiveData._starVec ),
        toKernelArray( DActiveData._starTriMap ) ); 
    CudaCheckError();

    // Get active triangle number and triangle map
    const int activeTriNum = makeInPlaceMapAndSum( *DActiveData._starTriMap );

    if ( LogStats )
    {
        const int activeStarNum = DActiveData._starVec->size();
        cout << "Stars with insertion: " << activeStarNum << endl;

        LogSplayInsNum += ( int ) DInsertData._vertVec->size();

        const int insSum    = thrust::reduce( DStarData._insCountVec->begin(), DStarData._insCountVec->end() );
        const int insAvg    = insSum / activeStarNum;

        cout << "Average insertions per star: " << insAvg << endl;
    }

    __copyInsertionsToHistory();

    // Store triangle index of active triangles for reuse
    TriPositionDVec& activeTriPosVec = *DTriBufVec;
    activeTriPosVec.resize( activeTriNum );
    ShortDVec activeTriInsNumVec( activeTriNum );

    // Find insertion count per triangle of active stars
    updateActiveTriData( activeTriPosVec, activeTriInsNumVec );

    ////
    // Insert points to stars
    ////

    DBeneathData._flagVec->fill( 0 );
    DBeneathData._beneathTriPosVec->fill( -1 );

    if ( LogStats )
    {
        cout << "activeStar = " << DActiveData._starVec->size() << ", activeTri = " << activeTriNum << endl;
    }

    for ( int insIdx = 0; insIdx < InsertPointMax; ++insIdx )
    {
        __addOnePointToStar( activeTriPosVec, activeTriInsNumVec, insIdx );
    }

    kerCountPointsOfStar<<< BlocksPerGrid, ThreadsPerBlock >>>( DStarData.toKernel() ); 
    CudaCheckError();

    return;
}

void _insertPointsToStars()
{
    if ( LogTiming )
    {
        insertTimer.start();
    }

    __insertPointsToStars();

    if ( LogTiming )
    {
        insertTimer.stop();
        insertTime += insertTimer.value();
    }

    return;
}

void areStarsConsistent()
{
    if ( !DoCheck )
    {
        return;
    }

    // Clear flags
    DBeneathData._flagVec->fill( 0 );

    kerCheckStarConsistency<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        DBeneathData.toKernel() );
    CudaCheckError();

    if ( 0 == (*DBeneathData._flagVec)[ ExactTriCount ] )
    {
        cout << "Stars are consistent!" << endl;
    }
    else
    {
        cout << "Star are NOT consistent!!!" << endl;
    }

    return;
}

void processFacets()
{
    LoopNum = 0;

    do
    {
        bool isSplayingDone = false; 

        if ( LogVerbose )
        {
            cout << endl << "Loop: " << LoopNum << endl;
        }

        if ( LogTiming )
        {
            loopTimer.start();
        }

        ////
        // Splay
        ////

        _getDrownedInsertions();
        const int insertNum = _gatherInsertions();

        if ( 0 == insertNum )
        {
            isSplayingDone = true; 
        }
        else
        {
            _sortInsertions();
            _expandStarsForInsertion();
            _insertPointsToStars();
        }

        ////
        // Finish splaying
        ////

        if ( isSplayingDone )
        {
            areStarsConsistent();
            break;
        }

        if ( LogTiming )
        {
            loopTimer.stop();

            char loopStr[10];
            sprintf( loopStr, "Loop %d", LoopNum );
            loopTimer.print( loopStr );
        }

        ++LoopNum;

    } while ( true );

    if ( LogStats )
    {
        cout << endl;
        cout << "Total splay insertions: " << LogSplayInsNum << endl;

        LogLoopNum = LoopNum;
    }

    if ( LogTiming )
    {
        cout << endl;
        cout << "Drowned time:    " << drownTime << endl;
        cout << "Get insert time: " << getInsertTime << endl;
        cout << "Expand time:     " << expandTime << endl;
        cout << "Sort time:       " << sortTime << endl;
        cout << "Insert time:     " << insertTime << endl;
    }

    // No longer needed
    DInsertData.deInit();

    return;
}

////////////////////////////////////////////////////////////////////////////////
//                           makeTetraFromStars()
////////////////////////////////////////////////////////////////////////////////

void _doHistoryStats()
{
    // History statistics
    if ( LogStats )
    {
        IntHVec histMap[2];
        IntHVec vertVec[2];

        for ( int i = 0; i < 2; ++i )
        {
            DHistoryData._starVertMap[i]->copyToHost( histMap[i] );
            DHistoryData._vertVec[i]->copyToHost( vertVec[i] );
        }

        const int starNum   = histMap[0].size();
        int maxVert         = 0;

        // Iterate history of each star
        for ( int star = 0; star < starNum; ++star )
        {
            int totVertNum = 0;

            for ( int hi = 0; hi < 2; ++hi )
            {
                const int vertBeg = histMap[ hi ][ star ];
                const int vertEnd = ( ( star + 1 ) < starNum ) ? histMap[ hi ][ star + 1 ] : vertVec[hi].size();
                const int vertNum = vertEnd - vertBeg;

                totVertNum += vertNum;
            }

            if ( totVertNum > maxVert )
            {
                maxVert = totVertNum;
            }
        }

        cout << "Total history: " << vertVec[0].size() + vertVec[1].size() << endl;
        cout << "Max:           " << maxVert << endl << endl;
    }

    return;
}

void _gatherTetraFromStars()
{
    _doHistoryStats();

    ////
    // Make tetra-tri map by looking at owner triangles
    ////

    const int triNum    = DStarData._triData.totalSize();
    IntDVec& ownTriVec  = *DTriBufVec;

    ownTriVec.resize( triNum, -1 );

    kerNoteOwnerTriangles<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        toKernelArray( ownTriVec ) );
    CudaCheckError();

    // Both lower- and upper-hull tetra are here
    const int lowUpTetraNum = compactIfNegative( ownTriVec );

    // lowUpTetraNum <<< triNum, so copy to new array
    IntDVec tetraTriMap;
    tetraTriMap.copyFrom( ownTriVec );

    ////
    // Mark and keep lower-hull tetra info
    ////

    kerMarkLowerHullTetra<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DPredicateInfo,
        DPointData.toKernel(),
        DStarData.toKernel(),
        toKernelArray( tetraTriMap ) );
    CudaCheckError();

    const int tetraNum = compactIfNegative( tetraTriMap );

    ////
    // Create facets for tetra
    ////

    IntDVec& triTetraMap = *DTriBufVec; // Reuse above array

    triTetraMap.resize( triNum, -1 );
    IntDVec facetStarVec( tetraNum );
    IntDVec facetTriVec( tetraNum );

    kerMakeCloneFacets<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        toKernelArray( tetraTriMap ),
        toKernelPtr( triTetraMap ),
        toKernelPtr( facetStarVec ),
        toKernelPtr( facetTriVec ) );
    CudaCheckError();

    ////
    // Get clone triangle info
    ////

    thrust::sort_by_key( facetStarVec.begin(), facetStarVec.end(), facetTriVec.begin() );

    LocTriIndexDVec tetraCloneTriVec( tetraNum );

    kerGetCloneTriInfo<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        toKernelArray( facetStarVec ),
        toKernelPtr( facetTriVec ),
        toKernelPtr( triTetraMap ),
        toKernelPtr( tetraCloneTriVec ) );
    CudaCheckError();

    ////
    // Create final tetra and its adjacencies
    ////

    DTetraData._vec->resize( tetraNum );

    kerGrabTetrasFromStars<<< BlocksPerGrid, ThreadsPerBlock >>>(
        DStarData.toKernel(),
        DTetraData.toKernel(),
        toKernelArray( tetraTriMap ),
        toKernelPtr( triTetraMap ),
        toKernelPtr( tetraCloneTriVec ) );
    CudaCheckError();

    return;
}

void _copyTetraToHost()
{
    HTetraVec.clear(); 
    DTetraData._vec->copyToHost( HTetraVec ); 

    return;
}

void makeTetraFromStars()
{
    _gatherTetraFromStars();
    _copyTetraToHost();

    return;
}

const TetraHVec& getHostTetra()
{
    return HTetraVec;
}

////////////////////////////////////////////////////////////////////////////////
