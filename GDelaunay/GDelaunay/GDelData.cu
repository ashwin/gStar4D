/*
Author: Ashwin Nanjappa
Filename: GDelData.cu

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
//                            GDelaunay Global Data
////////////////////////////////////////////////////////////////////////////////

// Self
#include "GDelData.h"

// Project
#include "Config.h"
#include "GDelKernels.h"

// Externs
extern int ThreadsPerBlock;
extern int BlocksPerGrid;

/////////////////////////////////////////////////////////////////// PointData //

struct GetMortonNumber
{
    // Note: No performance benefit by changing by-reference to by-value here
    // Note: No benefit by making this __forceinline__
    __device__ int operator () ( const Point3& point ) const
    {
        const int Guard = 0xFFFFFC00;   // 22 1-bits, 10 0-bits
        const int Gap16 = 0x030000FF;   // Creates 16-bit gap between value bits
        const int Gap08 = 0x0300F00F;   // ... and so on ...
        const int Gap04 = 0x030C30C3;   // ...
        const int Gap02 = 0x09249249;   // ...

        int coord[3] = { ( int ) point._p[0], ( int ) point._p[1], ( int ) point._p[2] };

        // Iterate coordinates of point
        for ( int vi = 0; vi < 3; ++vi )
        {
            // Read
            int v = coord[ vi ];

            CudaAssert( ( 0 == ( v & Guard ) ) && "Coordinate value is negative OR occupies more than 10 bits!" );

            // Create 2-bit gaps between the 10 value bits
            // Ex: 1001001001001001001001001001
            v = ( v | ( v << 16 ) ) & Gap16;
            v = ( v | ( v <<  8 ) ) & Gap08;
            v = ( v | ( v <<  4 ) ) & Gap04;
            v = ( v | ( v <<  2 ) ) & Gap02;

            // Write back
            coord[ vi ] = v;
        }

        // Interleave bits of x-y-z coordinates
        const int mortonNum = ( coord[ 0 ] | ( coord[ 1 ] << 1 ) | ( coord[ 2 ] << 2 ) );

        return mortonNum;
    }
};

void PointData::init( Point3HVec& pointHVec, Point3HVec& scaledHVec )
{
    _pointVec       = new Point3DVec( pointHVec );
    _scaledVec      = scaledHVec.empty() ? NULL : ( new Point3DVec( scaledHVec ) );
    _bitsPerIndex   = ( int ) ceil( log( ( double ) _pointVec->size() ) / log( 2.0 ) ); // Number of bits to store point index

    if ( getConfig()._doSorting )
    {
        ////
        // Sort points by Morton order
        ////

        // Space for Morton number of each point
        IntDVec orderVec( _pointVec->size() );

        // Pick vector with points inside scaled cube
        const Point3DVec* morPointVec = ( NULL == _scaledVec ) ? _pointVec : _scaledVec;
        
        // Generate Morton number of each point
        thrust::transform( morPointVec->begin(), morPointVec->end(), orderVec.begin(), GetMortonNumber() );

        // Use Morton number to sort points on device
        if ( NULL == _scaledVec )
        {
            thrust::sort_by_key( orderVec.begin(), orderVec.end(), _pointVec->begin() );
        }
        else
        {
            thrust::sort_by_key(    orderVec.begin(), orderVec.end(),
                                    thrust::make_zip_iterator( make_tuple( _pointVec->begin(), _scaledVec->begin() ) ) );
        }

        // Copy back sorted points to host

        pointHVec = *_pointVec;

        if ( _scaledVec )
        {
            scaledHVec = *_scaledVec;
        }
    }

    return;
}

void PointData::deinit()
{
    safeDeleteDevConPtr( &_pointVec );
    safeDeleteDevConPtr( &_scaledVec );

    _bitsPerIndex = -1;

    return;
}

KerPointData PointData::toKernel()
{
    KerPointData pData;

    pData._pointArr = toKernelPtr( _pointVec );
    pData._num      = ( int ) _pointVec->size();

    return pData;
}

//////////////////////////////////////////////////////////////// TriangleData //

void TriangleData::init()
{
    for ( int i = 0; i < 2; ++i )
    {
        _triVec[i]          = new TriDVec();
        _triOppVec[i]       = new TriOppDVec();
        _triStarVec[i]      = new IntDVec();
        _triStatusVec[i]    = new TriStatusDVec();
    }

    return;
}

void TriangleData::deinit()
{
    for ( int i = 0; i < 2; ++i )
    {
        safeDeleteDevConPtr( &_triVec[i] );
        safeDeleteDevConPtr( &_triOppVec[i] );
        safeDeleteDevConPtr( &_triStarVec[i] );
        safeDeleteDevConPtr( &_triStatusVec[i] );
    }

    return;
}

void TriangleData::resize( int newSize, int arrId, const TriangleStatus& triStatus ) 
{
    _triVec[ arrId ]->resize( newSize );
    _triOppVec[ arrId ]->resize( newSize ); 
    _triStarVec[ arrId ]->resize( newSize ); 
    _triStatusVec[ arrId ]->resize( newSize, triStatus ); 

    return;
}

int TriangleData::size( int vecId ) const
{
    return _triVec[ vecId ]->size();
}

int TriangleData::totalSize() const
{
    return size( 0 ) + size( 1 );
}

//////////////////////////////////////////////////////////////////// StarData //

void StarData::init( int pointNum )
{
    // Preallocate these per-star vectors
    _starTriMap[0]  = new IntDVec( pointNum );
    _starTriMap[1]  = new IntDVec( pointNum );
    _pointNumVec    = new IntDVec( pointNum );
    _maxSizeVec     = new IntDVec( pointNum );
    _insCountVec    = new IntDVec( pointNum );

    _triData.init();

    return;
}

void StarData::deInit()
{
    _triData.deinit(); 

    safeDeleteDevConPtr( &_starTriMap[0] );
    safeDeleteDevConPtr( &_starTriMap[1] );
    safeDeleteDevConPtr( &_pointNumVec );
    safeDeleteDevConPtr( &_maxSizeVec );
    safeDeleteDevConPtr( &_insCountVec );

    return;
}

KerStarData StarData::toKernel()
{
    KerStarData sData;

    for ( int i = 0; i < 2; ++i )
    {
        sData._triNum[i]        = ( int ) _triData.size( i );
        sData._triArr[i]        = toKernelPtr( _triData._triVec[i] );
        sData._triOppArr[i]     = toKernelPtr( _triData._triOppVec[i] );
        sData._triStarArr[i]    = toKernelPtr( _triData._triStarVec[i] );
        sData._triStatusArr[i]  = toKernelPtr( _triData._triStatusVec[i] );
        sData._starTriMap[i]    = toKernelPtr( _starTriMap[i] );
    }

    sData._starNum      = _starNum;
    sData._totalTriNum  = sData._triNum[0] + sData._triNum[1];
    sData._pointNumArr  = toKernelPtr( _pointNumVec );
    sData._maxSizeArr   = toKernelPtr( _maxSizeVec );
    sData._insCountArr  = toKernelPtr( _insCountVec );

    return sData;
}

// We move all triangle arrays EXCEPT triStar, which should already be updated!
template< typename T >
__global__ void kerMoveTriangleArray
(
int             oldTriNum,
KerIntArray     oldNewMap,
KerArray< T >   oldArr,
KerArray< T >   newArr
)
{
    // Iterate through triangles
    for ( int oldTriIdx = getCurThreadIdx(); oldTriIdx < oldTriNum; oldTriIdx += getThreadNum() )
    {
        // Skip free triangles

        const int newTriIdx = oldNewMap._arr[ oldTriIdx ];

        if ( -1 == newTriIdx )
        {
            continue;
        }

        // Copy old to new 
        newArr._arr[ newTriIdx ] = oldArr._arr[ oldTriIdx ];
    }

    return;
}

// Expands input vector to hold old data
template< typename T >
void StarData::expandData( int oldSize, int newSize, IntDVec& oldNewMap, DeviceContainer< T >& inVec )
{
    DeviceContainer< T > tmpVec( newSize ); 

    if ( oldSize > 0 ) 
    {
        kerMoveTriangleArray<<< BlocksPerGrid, ThreadsPerBlock >>>(
            oldSize,
            toKernelArray( oldNewMap ),
            toKernelArray( inVec ),
            toKernelArray( tmpVec ) );
        CudaCheckError();
    }

    inVec.swap( tmpVec );

    return;
}

void StarData::expandTriangles( int newSize, IntDVec& newTriMap )
{
    const int oldSize = ( int ) _triData.size( 1 );    // Grab old size before it is replaced

    ////
    // Create old-to-new triangle index map
    // *and* also update triStar and triStatus
    ////

    IntDVec newStarVec( newSize );
    TriStatusDVec newStatusVec( newSize, Free );
    IntDVec oldNewMap( oldSize, -1 );

    if ( oldSize > 0 ) 
    {
        kerMakeOldToNewTriMap<<< BlocksPerGrid, ThreadsPerBlock >>>(
            toKernel(),
            oldSize,
            toKernelArray( newTriMap ),
            toKernelArray( oldNewMap ),
            toKernelArray( newStarVec ),
            toKernelArray( newStatusVec ) );
        CudaCheckError();
    }

    _starTriMap[1]->swap( newTriMap );
    _triData._triStarVec[1]->swapAndFree( newStarVec );
    _triData._triStatusVec[1]->swapAndFree( newStatusVec );

    // Move rest of triangle arrays
    expandData( oldSize, newSize, oldNewMap, *_triData._triVec[1] ); 
    expandData( oldSize, newSize, oldNewMap, *_triData._triOppVec[1] ); 

    return;
}

///////////////////////////////////////////////////////////////// MissingData //

void MissingData::init()
{
    _memberVec  = new IntDVec();
    _leaderVec  = new IntDVec();

    return;
}

void MissingData::deInit()
{
    safeDeleteDevConPtr( &_memberVec );
    safeDeleteDevConPtr( &_leaderVec );

    return;
}

KerMissingData MissingData::toKernel()
{
    KerMissingData mData;

    mData._memberArr    = toKernelPtr( _memberVec );
    mData._leaderArr    = toKernelPtr( _leaderVec );
    mData._num          = _memberVec->size();

    return mData;
}

/////////////////////////////////////////////////////////////// InsertionData //

void InsertionData::init()
{
    _vertVec        = new IntDVec();
    _vertStarVec    = new IntDVec();
    _starVertMap    = new IntDVec();
    _shelveStarVec  = new IntDVec();
    _shelveVertVec  = new IntDVec();

    return;
}

void InsertionData::deInit() 
{
    safeDeleteDevConPtr( &_vertVec );
    safeDeleteDevConPtr( &_vertStarVec );
    safeDeleteDevConPtr( &_starVertMap );
    safeDeleteDevConPtr( &_shelveStarVec );
    safeDeleteDevConPtr( &_shelveVertVec );

    return;
}

KerInsertData InsertionData::toKernel()
{
    KerInsertData iData;

    iData._vertArr      = toKernelPtr( _vertVec );
    iData._vertStarArr  = toKernelPtr( _vertStarVec );
    iData._starVertMap  = toKernelPtr( _starVertMap );
    iData._vertNum      = _vertVec->size();
    iData._starNum      = _starVertMap->size();

    return iData;
}

///////////////////////////////////////////////////////////////// HistoryData //

void HistoryData::init()
{
    for ( int i = 0; i < 2; ++i )
    {
        _vertVec[ i ]       = new IntDVec();
        _vertStarVec[ i ]   = new IntDVec();
        _starVertMap[ i ]   = new IntDVec();
    }

    return;
}

void HistoryData::deInit()
{
    for ( int i = 0; i < 2; ++i )
    {
        safeDeleteDevConPtr( &_vertVec[ i ] );
        safeDeleteDevConPtr( &_starVertMap[ i ] );
        safeDeleteDevConPtr( &_vertStarVec[ i ] );
    }

    return;
}

KerHistoryData HistoryData::toKernel()
{
    KerHistoryData hData;

    for ( int i = 0; i < 2; ++i )
    {
        hData._vertArr[i]       = toKernelPtr( _vertVec[i] );
        hData._vertStarArr[i]   = toKernelPtr( _vertStarVec[i] );
        hData._starVertMap[i]   = toKernelPtr( _starVertMap[i] );
        hData._vertNum[i]       = _vertVec[i]->size();
    }

    return hData;
}

///////////////////////////////////////////////////////////////// BeneathData //

void BeneathData::init( int pointNum )
{
    _beneathTriPosVec   = new TriPositionDVec( pointNum );
    _exactTriPosVec     = new TriPositionDVec( ExactTriangleMax );
    _flagVec            = new IntDVec( FlagNum, 0 );

    return;
}

void BeneathData::deInit()
{
    safeDeleteDevConPtr( &_beneathTriPosVec );
    safeDeleteDevConPtr( &_exactTriPosVec );
    safeDeleteDevConPtr( &_flagVec );

    return;
}

KerBeneathData BeneathData::toKernel()
{
    KerBeneathData bData;

    bData._beneathTriPosArr = toKernelPtr( _beneathTriPosVec );
    bData._exactTriPosArr   = toKernelPtr( _exactTriPosVec );
    bData._flagArr          = toKernelPtr( _flagVec );

    return bData;
}

////////////////////////////////////////////////////////////////// ActiveData //

void ActiveData::init()
{
    _starVec        = new IntDVec();
    _starTriMap     = new IntDVec();

    return;
}

void ActiveData::deInit()
{
    safeDeleteDevConPtr( &_starVec );
    safeDeleteDevConPtr( &_starTriMap );

    return;
}

KerActiveData ActiveData::toKernel()
{
    KerActiveData aData;

    aData._starArr          = toKernelPtr( _starVec );
    aData._starTriMap       = toKernelPtr( _starTriMap );

    return aData;
}

/////////////////////////////////////////////////////////////////// TetraData //

void TetraData::init()
{
    _vec = new TetraDVec();
    return;
}

void TetraData::deInit() 
{
    safeDeleteDevConPtr( &_vec );
    return;
}

KerTetraData TetraData::toKernel()
{
    KerTetraData tData;

    tData._arr  = toKernelPtr( _vec );
    tData._num  = _vec->size();

    return tData;
}

////////////////////////////////////////////////////////////////////////////////
