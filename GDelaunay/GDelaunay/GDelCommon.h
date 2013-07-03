/*
Author: Ashwin Nanjappa
Filename: GDelCommon.h

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
//                          GDelaunay Utility
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Geometry.h"
#include "STLWrapper.h"
#include "ThrustWrapper.h"

// Launch configuration for complex kernels
#define MAX_THREADS_PER_BLOCK       256
#define MAX_PRED_THREADS_PER_BLOCK  64
#define MIN_BLOCKS_PER_MP           2

const int Marker                = -1;
const int MinWorksetSize        = 8; 
const int ProofPointsPerStar    = 4;
const int ExactTriangleMax      = 10240;
const int LookupBlockDim        = 256;  // Do NOT change!

///////////////////////////////////////////////////////////////// Basic Types //

template < typename T >
struct KerArray
{
    T*  _arr;
    int _num;
};

typedef unsigned char Byte;

typedef Byte TriangleStatus;

enum TriangleStatusValue
{
    TriangleStatusMin, // Always keep this at top!
    Valid,
    ValidAndUnchecked,
    NewValidAndUnchecked,
    Free,
    DoExactOnValid,
    DoExactOnUnchecked,
    TriangleStatusMax, // Always keep this at bottom!
};

__forceinline__ __host__ __device__ bool triNeedsExactCheck( TriangleStatus _status ) 
{
    return ( ( DoExactOnValid == _status ) || ( DoExactOnUnchecked == _status ) );
}

///////////////////////////////////////////////////////////////// TriPosition //

typedef int TriPosition;

struct TriPositionEx
{
    int _arrId;
    int _triIdx;
};

__forceinline__ __device__ TriPosition makeTriPos( int arrId, int triIdx )
{
    return ( ( triIdx << 1 ) + arrId ); 
}

__forceinline__ __device__ TriPositionEx makeTriPosEx( int arrId, int triIdx ) 
{
    TriPositionEx triPosEx; 

    triPosEx._arrId     = arrId; 
    triPosEx._triIdx    = triIdx; 

    return triPosEx;
}

__forceinline__ __device__ TriPositionEx triPosToEx( const TriPosition triPos ) 
{
    TriPositionEx triPosEx; 

    triPosEx._arrId     = ( triPos & 1 ); 
    triPosEx._triIdx    = ( triPos >> 1 ); 

    return triPosEx;
}

__forceinline__ __device__ TriPosition exToTriPos( const TriPositionEx triPosEx )
{
    return ( ( triPosEx._triIdx << 1 ) + triPosEx._arrId ); 
}

///////////////////////////////////////////////////////////////// Memory Pool //

struct Pool
{
    int                             _blockSize;
    thrust::host_vector< void* >    _memory; 

    void init( int );
    void deInit();
};

Pool& getPool();

///////////////////////////////////////////////////////////// DeviceContainer //

template< typename T > 
class DeviceContainer
{
private:
    thrust::device_ptr< T > _ptr; 
    int                     _size; 
    int                     _capacity; 

public: 
    typedef thrust::device_ptr< T > iterator; 

    DeviceContainer( ) : _size( 0 ), _capacity( 0 ) { }; 
    
    DeviceContainer( int n ) : _size( 0 ), _capacity( 0 )
    {
        resize( n ); 
        return;
    }

    DeviceContainer( int n, T value ) : _size( 0 ), _capacity( 0 )
    {
        resize( n, value );
        return;
    }

    ~DeviceContainer()
    {
        free();
        return;
    }

    void free() 
    {
        if ( _capacity > 0 )
        {
            if ( getPool()._blockSize == ( _capacity * sizeof( T ) ) )
            {
                getPool()._memory.push_back( _ptr.get() ); 
            }
            else 
            {
                CudaSafeCall( cudaFree( _ptr.get() ) );
            }
        }

        _size       = 0; 
        _capacity   = 0; 

        return;
    }

    // Use only for cases where new size is within capacity
    // So, old data remains in-place
    void expand( int n )
    {
        assert( ( _capacity >= n ) && "New size not within current capacity! Use resize!" );

        _size = n;

        return;
    }

    void resize( int n )
    {
        if ( _capacity >= n )
        {
            _size = n; 
            return;
        }

        free(); 

        _size       = n; 
        _capacity   = ( n == 0 ) ? 1 : n; 

        if ( ( _capacity * sizeof( T ) == getPool()._blockSize ) &&
            getPool()._memory.size() > 0 ) 
        {
            _ptr = thrust::device_ptr< T >( ( T* ) getPool()._memory.back() );
            getPool()._memory.pop_back(); 
        }
        else
        {
            try
            {
                _ptr = thrust::device_malloc< T >( _capacity );
            }
            catch( ... )
            {
                // output an error message and exit
                const int OneMB = ( 1 << 20 );
                cerr << "thrust::device_malloc failed to allocate " << ( sizeof( T ) * _capacity ) / OneMB << " MB!" << endl;
                cuPrintMemory( "" );
                exit( -1 );
            }
        }

        return;
    }

    void resize( int n, const T& value )
    {
        resize( n ); 
        thrust::fill_n( begin(), n, value );
        return;
    }

    int size() const { return _size; }
    int capacity() const { return _capacity; }

    thrust::device_reference< T > operator[] ( const int index ) const
    {
        return _ptr[ index ]; 
    }

    const iterator begin() const { return _ptr; }

    const iterator end() const { return _ptr + _size; }

    void erase( const iterator& first, const iterator& last )
    {
        if ( last == end() )
        {
            _size -= (last - first);
        }
        else
        {
            assert( false && "Not supported right now!" );
        }

        return;
    }

    void swap( DeviceContainer< T >& arr ) 
    {
        int tempSize    = _size; 
        int tempCap     = _capacity; 
        T* tempPtr      = ( _capacity > 0 ) ? _ptr.get() : 0; 

        _size       = arr._size; 
        _capacity   = arr._capacity; 

        if ( _capacity > 0 )
        {
            _ptr = thrust::device_ptr< T >( arr._ptr.get() ); 
        }

        arr._size       = tempSize; 
        arr._capacity   = tempCap; 

        if ( tempCap >= 0 )
        {
            arr._ptr = thrust::device_ptr< T >( tempPtr );
        }

        return;
    }
    
    // Input array is freed
    void swapAndFree( DeviceContainer< T >& inArr )
    {
        swap( inArr );
        inArr.free();
        return;
    }

    void copyFrom( const DeviceContainer< T >& inArr )
    {
        resize( inArr.size() );
        thrust::copy( inArr.begin(), inArr.end(), begin() );
        return;
    }

    void fill( const T& value )
    {
        thrust::fill_n( _ptr, _size, value );
        return;
    }

    void copyToHost( thrust::host_vector< T >& dest )
    {
        dest.insert( dest.begin(), begin(), end() );
        return;
    }

    // Do NOT remove! Useful for debugging.
    void copyFromHost( const thrust::host_vector< T >& inArr )
    {
        resize( inArr.size() );
        thrust::copy( inArr.begin(), inArr.end(), begin() );
        return;
    }

    DeviceContainer& operator=( DeviceContainer& src )
    {
        resize( src._size ); 

        if ( src._size > 0 )
        {
            thrust::copy( src.begin(), src.end(), begin() ); 
        }

        return *this; 
    }
}; 

// Delete DeviceContainer pointer safely
template < typename T >
void safeDeleteDevConPtr( T** ptr )
{
    if ( *ptr )
    {
        delete *ptr;
        *ptr = NULL;
    }

    CudaCheckError();

    return;
}

////////////////////////////////////////////////////////////////// DVec Types //

typedef DeviceContainer< char >             CharDVec;
typedef DeviceContainer< Byte >             ByteDVec;
typedef DeviceContainer< short >            ShortDVec;
typedef DeviceContainer< int >              IntDVec;
typedef DeviceContainer< Triangle >         TriDVec;
typedef DeviceContainer< TriangleOpp >      TriOppDVec;
typedef DeviceContainer< TriangleStatus >   TriStatusDVec;
typedef DeviceContainer< Tetrahedron >      TetraDVec;
typedef DeviceContainer< TriPosition >      TriPositionDVec;
typedef DeviceContainer< LocTriIndex >      LocTriIndexDVec;

typedef thrust::host_vector< Byte >             ByteHVec;
typedef thrust::host_vector< TriangleStatus >   TriStatusHVec;

typedef CharDVec::iterator                      CharDIter;
typedef IntDVec::iterator                       IntDIter;
typedef thrust::tuple< int, int >               IntTuple2;
typedef thrust::tuple< IntDIter, IntDIter >     IntDIterTuple2;
typedef thrust::zip_iterator< IntDIterTuple2 >  ZipDIter;

////////////////////////////////////////////////////////// ToKernel Functions //

template < typename T >
T* toKernelPtr( thrust::device_vector< T >& dVec )
{
    return thrust::raw_pointer_cast( &dVec[0] );
}

template < typename T >
T* toKernelPtr( thrust::device_vector< T >* dVec )
{
    return thrust::raw_pointer_cast( &(*dVec)[0] );
}

template < typename T >
T* toKernelPtr( DeviceContainer< T >& dVec )
{
    return thrust::raw_pointer_cast( &dVec[0] );
}

template < typename T >
T* toKernelPtr( DeviceContainer< T >* dVec )
{
    return thrust::raw_pointer_cast( &(*dVec)[0] );
}

template < typename T >
KerArray< T > toKernelArray( DeviceContainer< T >& dVec )
{
    KerArray< T > tArray;
    tArray._arr = toKernelPtr( dVec );
    tArray._num = dVec.size();

    return tArray;
}

template < typename T >
KerArray< T > toKernelArray( DeviceContainer< T >* dVec )
{
    KerArray< T > tArray;
    tArray._arr = toKernelPtr( dVec );
    tArray._num = dVec->size();

    return tArray;
}

//////////////////////////////////////////////////////////////////// Functors //

struct IsZero
{
    __host__ __device__ bool operator() ( const int x )
    {
        return ( x == 0 );
    }
};

struct IsNegative
{
    __host__ __device__ bool operator() ( const int x )
    {
        return ( x < 0 );
    }
};

struct GetPumpedWorksetSize
{
    __host__ __device__ int operator() ( int worksetSize )
    {
        return ( worksetSize < MinWorksetSize ) ? ( MinWorksetSize - worksetSize ) : 0;
    }
};

struct FlipNegToPos
{
    __host__ __device__ int operator() ( int val )
    {
        return ( val < 0 ) ? ( - val - 1 ) : val;
    }
};

template < typename T >
int compactIfZero( DeviceContainer< T >& inVec, const IntDVec& checkVec )
{
    assert( ( inVec.size() == checkVec.size() ) && "Vectors should be equal size!" );

    inVec.erase(    thrust::remove_if( inVec.begin(), inVec.end(), checkVec.begin(), IsZero() ),
                    inVec.end() );

    return ( int ) inVec.size();
}

// Remove negative elements in input vector
template < typename T >
int compactIfNegative( DeviceContainer< T >& inVec )
{
    inVec.erase(    thrust::remove_if( inVec.begin(), inVec.end(), IsNegative() ),
                    inVec.end() );

    return ( int ) inVec.size();
}

// Convert input vector to map and also calculate the sum of input array
// Input:  [ 4 2 0 5 ]
// Output: [ 0 4 6 6 ] Sum: 11
template< typename T >
int makeMapAndSum( const DeviceContainer< T >& inVec, IntDVec& mapVec )
{
    // Resize map vector
    mapVec.resize( inVec.size() );

    // Make map
    thrust::exclusive_scan( inVec.begin(), inVec.end(), mapVec.begin() );

    // Sum
    const int sum = inVec[ inVec.size() - 1 ] + mapVec[ mapVec.size() - 1 ];

    return sum;
}

void makeAllStarMap( IntDVec& inVec, IntDVec& mapVec, int starNum );
void convertMapToCountVec( IntDVec& inMap, IntDVec& countVec, int dataNum );
void makeAllStarCountVec( IntDVec& dataVec, IntDVec& countVec, int starNum );
void convertAllStarMapToSimpleMap( IntDVec& inMap, int dataNum );
int makeInPlaceMapAndSum( IntDVec& inVec );
void compactBothIfEqual( IntDVec& vec0, IntDVec& vec1 );
void compactBothIfZero( IntDVec& vec0, IntDVec& vec1 );
void compactBothIfNegative( IntDVec& vec0, IntDVec& vec1 );
void compactBothIfNegative( IntDVec& vec0, IntDVec& vec1, int begin );
void makePairVectorUnique( IntDVec& vec0, IntDVec& vec1 );
void sortAndUniqueUsingKeyValAppend( IntDVec& keyVec, IntDVec& valVec, int bitsPerIndex );

///////////////////////////////////////////////////////////////////// Bitmask //

__forceinline__ __host__ __device__ int getLookupPaddedNum( int num )
{
    return ( ( num + LookupBlockDim - 1 ) / LookupBlockDim ) * LookupBlockDim;
}

__forceinline__ __host__ __device__ int getLookupBlockIdx( int idx )
{
    return ( idx / LookupBlockDim );
}

__forceinline__ __host__ __device__ int getByteIdx( int idx )
{
    return ( idx >> 3 ); // Div by 8
}

__forceinline__ __host__ __device__ Byte getBitIdx( int idx )
{
    return ( idx & 0x7 ); // Mod by 8
}

__forceinline__ __host__ __device__ Byte getBit( Byte byteVal, Byte bitIdx )
{
    const Byte mask     = ( 0x1 << bitIdx );
    const Byte bitVal   = ( byteVal & mask ) >> bitIdx;

    return bitVal;
}

// Set bit in byte to 1
__forceinline__ __host__ __device__ Byte setBit( Byte byteVal, Byte bitIdx )
{
    const Byte mask     = ( 0x1 << bitIdx );
    const Byte newVal   = ( byteVal | mask );

    return newVal;
}

/////////////////////////////////////////////////////////////////////// Types //

enum FlagStatus
{
    ExactTriCount,
    FlagNum,    // Note: Keep this as last one
};

struct KerPointData
{
    Point3* _pointArr;
    int     _num;
};

struct KerTetraData
{
    Tetrahedron*  _arr;
    int           _num;
};

struct KerHistoryData
{
    int*    _vertArr[2];
    int*    _vertStarArr[2];
    int*    _starVertMap[2];
    int     _vertNum[2];
};

struct KerActiveData
{
    int*    _starArr;
    int*    _starTriMap;
    Byte*   _triBitmaskArr;
    int*    _triBlockMap;
    Byte*   _triPrefixArr;
};

struct KerBeneathData
{
    TriPosition*    _beneathTriPosArr;
    TriPosition*    _exactTriPosArr;
    int*            _flagArr;
};

////////////////////////////////////////////////////////////////////////////////

// Indices and sizes of triangles of a star
struct StarInfo
{
    int _begIdx0;
    int _size0;
    int _begIdx1MinusSize0;    
    int _locTriNum;
    int _triArrSize0;

    __forceinline__ __device__ int toGlobalTriIdx( int locTriIdx ) const
    {
        return ( locTriIdx < _size0 ) ? _begIdx0 + locTriIdx : _triArrSize0 + _begIdx1MinusSize0 + locTriIdx; 
    }

    __forceinline__ __device__ void moveToNextTri( TriPositionEx& triPosEx ) const
    {
        ++triPosEx._triIdx;

        // Incremented location is end of array0
        if ( ( 0 == triPosEx._arrId ) & ( triPosEx._triIdx == ( _begIdx0 + _size0 ) ) )
        {
            triPosEx._triIdx    = _begIdx1MinusSize0 + _size0; 
            triPosEx._arrId     = 1; 
        }

        return;
    }

    __forceinline__ __device__ LocTriIndex toLocTriIdx( TriPositionEx triPosEx ) const
    {
        return ( triPosEx._arrId == 0 ) ? ( triPosEx._triIdx - _begIdx0 ) : ( triPosEx._triIdx - _begIdx1MinusSize0 ); 
    }    

    __forceinline__ __device__ TriPositionEx locToTriPosEx( int locIdx ) const
    {
        TriPositionEx triPosEx; 

        triPosEx._arrId     = ( locIdx >= _size0 ); 
        triPosEx._triIdx    = ( locIdx >= _size0 ) ? _begIdx1MinusSize0 + locIdx : _begIdx0 + locIdx; 

        return triPosEx; 
    }

    __forceinline__ __device__ TriPosition locToTriPos( int locIdx ) const
    {
        const int arrId     = ( locIdx >= _size0 ); 
        const int triIdx    = ( locIdx >= _size0 ) ? _begIdx1MinusSize0 + locIdx : _begIdx0 + locIdx; 

        return makeTriPos( arrId, triIdx );
    }
};

struct KerStarData
{
    int _starNum;
    int _totalTriNum;
    int _triNum[2];

    Triangle*       _triArr[2];
    TriangleOpp*    _triOppArr[2];
    int*            _triStarArr[2]; 
    TriangleStatus* _triStatusArr[2];
    int*            _starTriMap[2];
    int*            _pointNumArr;
    int*            _maxSizeArr; 
    int*            _insCountArr;

    __forceinline__ __device__ StarInfo getStarInfo( int star ) const
    {
        const int triIdxBeg0    = _starTriMap[0][ star ];
        const int triIdxBeg1    = _starTriMap[1][ star ]; 

        const int triIdxEnd0    = ( star < ( _starNum - 1 ) ) ? _starTriMap[0][ star + 1 ] : _triNum[0];
        const int triIdxEnd1    = ( star < ( _starNum - 1 ) ) ? _starTriMap[1][ star + 1 ] : _triNum[1];

        CudaAssert( ( -1 != triIdxBeg0 ) && ( -1 != triIdxBeg1 ) && ( -1 != triIdxEnd0 ) && ( -1 != triIdxEnd1 ) ); 

        StarInfo starInfo; 

        starInfo._begIdx0           = triIdxBeg0;
        starInfo._size0             = triIdxEnd0 - triIdxBeg0;
        starInfo._begIdx1MinusSize0 = triIdxBeg1 - starInfo._size0;
        starInfo._locTriNum         = ( triIdxEnd0 - triIdxBeg0 ) + ( triIdxEnd1 - triIdxBeg1 );
        starInfo._triArrSize0       = _triNum[0]; 

        return starInfo; 
    }

    __forceinline__ __device__ TriPositionEx globToTriPosEx( int idx ) const
    {
        CudaAssert( ( idx >= 0 ) && ( idx < _totalTriNum ) && "Invalid index!" );

        TriPositionEx triPosEx; 

        triPosEx._arrId     = ( idx >= _triNum[0] );        
        triPosEx._triIdx    = idx - ( _triNum[0] * triPosEx._arrId );

        return triPosEx; 
    }

    __forceinline__ __device__ Triangle& triangleAt( TriPositionEx loc )
    {
        return _triArr[ loc._arrId ][ loc._triIdx ];
    }

    __forceinline__ __device__ TriangleOpp& triOppAt( TriPositionEx loc )
    {
        return _triOppArr[ loc._arrId ][ loc._triIdx ];
    }

    __forceinline__ __device__ int& triStarAt( TriPositionEx loc )
    {
        return _triStarArr[ loc._arrId ][ loc._triIdx ];
    }

    __forceinline__ __device__ TriangleStatus& triStatusAt( TriPositionEx loc )
    {
        return _triStatusArr[ loc._arrId ][ loc._triIdx ];
    }
};

struct PredicateInfo
{
    RealType*   _consts;
    RealType*   _data;

    void init()
    {
        _consts = NULL;
        _data   = NULL;

        return;
    }

    void deInit()
    {
        cuDelete( &_consts );
        cuDelete( &_data );

        return;
    }
};

struct KerMissingData
{
    int*    _memberArr;
    int*    _leaderArr;
    int     _num;
};

struct KerInsertData
{
    int* _vertArr;
    int* _vertStarArr;
    int* _starVertMap;

    int  _starNum;
    int  _vertNum;
};

struct KerDrownedData
{
    int* _arr;
    int* _mapArr;
    int* _indexArr;
    int  _num;
};

template < typename T >
KerArray< T > toKernelArray( thrust::device_vector< T >& dVec )
{
    KerArray< T > tArray;
    tArray._arr = thrust::raw_pointer_cast( &dVec[0] );
    tArray._num = ( int ) dVec.size();

    return tArray;
}

typedef KerArray< Byte >            KerByteArray;
typedef KerArray< short >           KerShortArray;
typedef KerArray< int >             KerIntArray;
typedef KerArray< Triangle >        KerTriangleArray;
typedef KerArray< TriangleStatus >  KerTriStatusArray;
typedef KerArray< TriPosition >     KerTriPosArray;

////////////////////////////////////////////////////////////////////////////////
