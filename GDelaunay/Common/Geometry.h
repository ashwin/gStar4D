/*
Author: Ashwin Nanjappa
Filename: Geometry.h

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
//                                Geometry
////////////////////////////////////////////////////////////////////////////////

#pragma once

// Project
#include "STLWrapper.h"
#include "CudaWrapper.h"
#include "ThrustWrapper.h"

enum Orient
{
    OrientNeg   = -1,
    OrientZero  = +0,
    OrientPos   = +1
};

/////////////////////////////////////////////////////////////////////// Point //

#ifdef REAL_TYPE_DOUBLE
typedef double RealType;
#else
typedef float RealType;
#endif

struct Point3
{
    RealType _p[ 3 ];

    bool lessThan( const Point3& ) const;
    bool operator < ( const Point3& ) const;
};

typedef thrust::host_vector< Point3 >   Point3HVec;
typedef thrust::device_vector< Point3 > Point3DVec;
typedef std::set< Point3 >              Point3Set;

///////////////////////////////////////////////////////////////////// Segment //

struct Segment
{
    int _v[2];

    __host__ __device__ bool equal ( const Segment& ) const;
    __host__ __device__ bool lessThan( const Segment& ) const;
    __host__ __device__ bool operator == ( const Segment& ) const;
    __host__ __device__ bool operator < ( const Segment& ) const;
};

typedef thrust::host_vector< Segment >  SegmentHVec;
typedef std::set< Segment >             SegmentSet;

//////////////////////////////////////////////////////////////////// Triangle //

typedef short LocTriIndex;

struct Triangle
{
    int _v[3];

    __host__ __device__ bool equal( const Triangle& ) const;
    __host__ __device__ bool lessThan( const Triangle& ) const;
    __host__ __device__ bool operator == ( const Triangle& ) const;
    __host__ __device__ bool operator < ( const Triangle& ) const;

    __device__ bool hasVertex( int vert ) const
    {
        return ( ( _v[0] == vert ) | ( _v[1] == vert ) | ( _v[2] == vert ) );
    }

    __device__ int indexOfVert( int vert ) const
    {
        CudaAssert( hasVertex( vert ) && "Vertex not in Triangle!" );

        return ( ( _v[1] == vert ) | ( ( _v[2] == vert ) << 1 ) );
    }

    __device__ int minus( const Triangle& tri ) const
    {
        if ( !tri.hasVertex( _v[0] ) ) return _v[0];
        if ( !tri.hasVertex( _v[1] ) ) return _v[1];
        
        return _v[2];       // Must have one vertex in common
    }
};

// Stores local triangle index of 3 opposite triangles
// and also index of this triangle in the 3 opposite triangles
struct TriangleOpp
{
    LocTriIndex  _opp[3];

    __forceinline__ __device__ void setOpp( int vi, LocTriIndex oppVal, int oppVi ) 
    {
        _opp[vi] = ( oppVal << 2 ) | oppVi; 
    }

    __forceinline__ __device__ LocTriIndex getOppTri( int vi ) const
    {
        return ( _opp[vi] >> 2 ); 
    }

    __forceinline__ __device__ int getOppVi( int vi ) const
    {
        return ( _opp[vi] & 3 ); 
    }
};

typedef thrust::host_vector< Triangle > TriangleHVec;
typedef std::set< Triangle >            TriangleSet;

///////////////////////////////////////////////////////////////// Tetrahedron //

struct Tetrahedron
{
    int _v[ 4 ];
    int _opp[ 4 ];

    __device__ bool hasVertex( int vert ) const
    {
        return ( ( _v[0] == vert ) || ( _v[1] == vert ) || ( _v[2] == vert ) || ( _v[3] == vert ) );
    }

    __device__ int indexOfVert( int vert ) const
    {
        CudaAssert( hasVertex( vert ) && "Vertex not in Tetrahedron!" );

        if      ( _v[0] == vert )   return 0;
        else if ( _v[1] == vert )   return 1;
        else if ( _v[2] == vert )   return 2;
        else                        return 3;
    }

    __device__ bool hasOpp( int oppVal ) const
    {
        return ( ( _opp[0] == oppVal ) || ( _opp[1] == oppVal ) || ( _opp[2] == oppVal ) || ( _opp[3] == oppVal ) );
    }

    __host__ __device__ int indexOfOpp( int oppVal ) const
    {
        CudaAssert( hasVertex( oppVal ) && "Opp not in Tetrahedron!" );

        if      ( _opp[0] == oppVal )   return 0;
        else if ( _opp[1] == oppVal )   return 1;
        else if ( _opp[2] == oppVal )   return 2;
        else                            return 3;
    }

    __device__ void setOpp( int vert, int oppTetIdx )
    {
        // indexOfVert() will assert
        const int idx   = indexOfVert( vert );
        _opp[ idx ]     = oppTetIdx;

        return;
    }

    void getSegments( Segment* ) const;
    void getTriangles( Triangle* ) const;
};

typedef thrust::host_vector< Tetrahedron >      TetraHVec;

class TetraMesh
{
public:
    void setPoints( const Point3HVec& );
    void setTetra( const TetraHVec& );
    void check();
    void writeToFile( const string& );

private:
    Point3HVec  _pointVec;
    TetraHVec   _tetraVec;

    void _checkEuler() const;
    int _getVertexCount() const;
    int _getSegmentCount() const;
    int _getTriangleCount() const;
    void _checkOrientation();
    void _checkInSphere();
};

////////////////////////////////////////////////////////////////////////////////
