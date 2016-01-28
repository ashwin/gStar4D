/*
Author: Ashwin Nanjappa
Filename: GDelCommon.cu

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
//                          GDelaunay Common
////////////////////////////////////////////////////////////////////////////////

// Self
#include "GDelCommon.h"

// Project
#include "Config.h"
#include "CudaWrapper.h"
#include "GDelKernels.h"

// Externs
extern int ThreadsPerBlock;
extern int BlocksPerGrid;

///////////////////////////////////////////////////////////////// Memory Pool //

void Pool::init( int blockSize ) 
{
    _blockSize = blockSize; 
    return;
}

void Pool::deInit()
{
    if ( getConfig()._logStats )
    {
        cout << endl << "Peak pool size: " << _memory.size() << endl; 
    }

    for ( int i = 0; i < ( int ) _memory.size(); ++i ) 
        CudaSafeCall( cudaFree( _memory[i] ) ); 

    _memory.clear(); 

    _blockSize = -1; 

    return;
}

Pool& getPool()
{
    static Pool _pool;
    return _pool;
}

////////////////////////////////////////////////////////////////////////////////

// Replace input vector with its map and also calculate the sum of input vector
// Input:  [ 4 2 0 5 ]
// Output: [ 0 4 6 6 ] Sum: 11
int makeInPlaceMapAndSum( IntDVec& inVec )
{
    const int lastValue = inVec[ inVec.size() - 1 ]; 

    // Make map
    thrust::exclusive_scan( inVec.begin(), inVec.end(), inVec.begin() );

    // Sum
    const int sum = inVec[ inVec.size() - 1 ] + lastValue; 

    return sum;
}

// Given an input list of sorted stars (with duplicates and missing stars)
// creates a map for all stars
void makeAllStarMap( IntDVec& inVec, IntDVec& mapVec, int starNum )
{
    // Expand map to input vector size
    mapVec.resize( starNum, -1 );

    kerMakeAllStarMap<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( inVec ),
        toKernelArray( mapVec ),
        starNum );
    CudaCheckError();

    return;
}

void convertMapToCountVec( IntDVec& inMap, IntDVec& countVec, int dataNum )
{
    countVec.resize( inMap.size() );

    kerConvertMapToCount<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( inMap ),
        toKernelPtr( countVec ),
        dataNum );
    CudaCheckError();

    return;
}

void makeAllStarCountVec( IntDVec& dataVec, IntDVec& countVec, int starNum )
{
    IntDVec mapVec;

    makeAllStarMap( dataVec, mapVec, starNum );
    convertMapToCountVec( mapVec, countVec, dataVec.size() );

    return;
}

// Input is overwritten
void convertAllStarMapToSimpleMap( IntDVec& inMap, int dataNum )
{
    IntDVec countVec;

    convertMapToCountVec( inMap, countVec, dataNum );

    thrust::exclusive_scan( countVec.begin(), countVec.end(), inMap.begin() );

    return;
}

// Check if both elements of tuple are equal
struct isTuple2Equal
{
    __host__ __device__ bool operator() ( const IntTuple2& tup )
    {
        const int x = thrust::get<0>( tup );
        const int y = thrust::get<1>( tup );

        return ( x == y );
    }
};

// Check if first value in tuple2 is negative
struct isTuple2Negative
{
    __host__ __device__ bool operator() ( const IntTuple2& tup )
    {
        const int x = thrust::get<0>( tup );
        return ( x < 0 );
    }
};

// Check if second value in tuple2 is zero
struct isTuple2Zero
{
    __host__ __device__ bool operator() ( const IntTuple2& tup )
    {
        const int x = thrust::get<0>( tup );
        return ( x == 0 );
    }
};

void compactBothIfZero( IntDVec& vec0, IntDVec& vec1 )
{
    assert( ( vec0.size() == vec1.size() ) && "Vectors should be equal size!" );

    ZipDIter newEnd = thrust::remove_if(    thrust::make_zip_iterator( thrust::make_tuple( vec0.begin(), vec1.begin() ) ),
                                            thrust::make_zip_iterator( thrust::make_tuple( vec0.end(), vec1.end() ) ),
                                            isTuple2Zero() );

    IntDIterTuple2 endTuple = newEnd.get_iterator_tuple();

    vec0.erase( thrust::get<0>( endTuple ), vec0.end() );
    vec1.erase( thrust::get<1>( endTuple ), vec1.end() );

    return;
}

void compactBothIfNegative( IntDVec& vec0, IntDVec& vec1 )
{
    assert( ( vec0.size() == vec1.size() ) && "Vectors should be equal size!" );

    ZipDIter newEnd = thrust::remove_if(    thrust::make_zip_iterator( thrust::make_tuple( vec0.begin(), vec1.begin() ) ),
                                            thrust::make_zip_iterator( thrust::make_tuple( vec0.end(), vec1.end() ) ),
                                            isTuple2Negative() );

    IntDIterTuple2 endTuple = newEnd.get_iterator_tuple();

    vec0.erase( thrust::get<0>( endTuple ), vec0.end() );
    vec1.erase( thrust::get<1>( endTuple ), vec1.end() );

    return;
}

void compactBothIfNegative( IntDVec& vec0, IntDVec& vec1, int offset )
{
    assert( ( vec0.size() == vec1.size() ) && "Vectors should be equal size!" );

    ZipDIter newEnd = thrust::remove_if(    thrust::make_zip_iterator( thrust::make_tuple( vec0.begin() + offset, vec1.begin() + offset ) ),
                                            thrust::make_zip_iterator( thrust::make_tuple( vec0.end(),           vec1.end() ) ),
                                            isTuple2Negative() );

    IntDIterTuple2 endTuple = newEnd.get_iterator_tuple();

    vec0.erase( thrust::get<0>( endTuple ), vec0.end() );
    vec1.erase( thrust::get<1>( endTuple ), vec1.end() );

    return;
}

void compactBothIfEqual( IntDVec& vec0, IntDVec& vec1 )
{
    assert( ( vec0.size() == vec1.size() ) && "Vectors should be equal size!" );

    ZipDIter newEnd = thrust::remove_if(    thrust::make_zip_iterator( thrust::make_tuple( vec0.begin(), vec1.begin() ) ),
                                            thrust::make_zip_iterator( thrust::make_tuple( vec0.end(), vec1.end() ) ),
                                            isTuple2Equal() );

    IntDIterTuple2 endTuple = newEnd.get_iterator_tuple();

    vec0.erase( thrust::get<0>( endTuple ), vec0.end() );
    vec1.erase( thrust::get<1>( endTuple ), vec1.end() );

    return;
}

// Remove duplicates from the input key-value vector pair
void makePairVectorUnique( IntDVec& vec0, IntDVec& vec1 )
{
    assert( ( vec0.size() == vec1.size() ) && "Invalid size vectors!" );

    kerMarkDuplicates<<< BlocksPerGrid, ThreadsPerBlock >>>( toKernelArray( vec0 ), toKernelArray( vec1 ) ); 
    CudaCheckError();

    compactBothIfNegative( vec1, vec0 ); 

    return;
}

// Sort by key and remove duplicates
void sortAndUniqueUsingKeyValAppend( IntDVec& keyVec, IntDVec& valVec, int bitsPerIndex )
{
    kerAppendValueToKey<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( keyVec ),
        toKernelPtr( valVec ),
        bitsPerIndex );
    CudaCheckError();

    thrust::sort_by_key( keyVec.begin(), keyVec.end(), valVec.begin() );
    
    makePairVectorUnique( keyVec, valVec ); // Remove duplicates from both vectors

    kerRemoveValueFromKey<<< BlocksPerGrid, ThreadsPerBlock >>>(
        toKernelArray( keyVec ),
        bitsPerIndex ); 
    CudaCheckError();

    return;
}

////////////////////////////////////////////////////////////////////////////////
