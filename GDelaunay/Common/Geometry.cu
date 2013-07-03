/*
Author: Ashwin Nanjappa
Filename: Geometry.cu

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

// Project
#include "Geometry.h"
#include "PerfTimer.h"

extern "C" {
    void exactinit();
    RealType orient3d( RealType *pa, RealType *pb, RealType *pc, RealType *pd );
    RealType insphere( RealType *pa, RealType *pb, RealType *pc, RealType *pd, RealType *pe );
}

/////////////////////////////////////////////////////////////////////// Point //

bool Point3::lessThan( const Point3& pt ) const
{
    if ( _p[0] < pt._p[0] )
        return true; 
    if ( _p[0] > pt._p[0] )
        return false; 
    if ( _p[1] < pt._p[1] )
        return true; 
    if ( _p[1] > pt._p[1] )
        return false; 
    if ( _p[2] < pt._p[2] )
        return true; 

    return false; 
}

bool Point3::operator < ( const Point3& pt ) const
{
    return lessThan( pt );
}

///////////////////////////////////////////////////////////////////// Segment //

__host__ __device__ bool Segment::equal( const Segment& seg ) const
{
    return ( ( _v[0] == seg._v[0] ) && ( _v[1] == seg._v[1] ) );
}

__host__ __device__ bool Segment::lessThan( const Segment& seg ) const
{
    if ( _v[0] < seg._v[0] )
        return true;
    if ( _v[0] > seg._v[0] )
        return false;
    if ( _v[1] < seg._v[1] )
        return true;

    return false; 
}

__host__ __device__ bool Segment::operator == ( const Segment& seg ) const
{
    return equal( seg );
}

__host__ __device__ bool Segment::operator < ( const Segment& seg ) const
{
    return lessThan( seg );
}

//////////////////////////////////////////////////////////////////// Triangle //

__host__ __device__ bool Triangle::equal( const Triangle& tri ) const
{
    return ( ( _v[0] == tri._v[0] ) && ( _v[1] == tri._v[1] ) && ( _v[2] == tri._v[2] ) );
}

__host__ __device__ bool Triangle::lessThan( const Triangle& tri ) const
{
    if ( _v[0] < tri._v[0] )
        return true; 
    if ( _v[0] > tri._v[0] )
        return false; 
    if ( _v[1] < tri._v[1] )
        return true; 
    if ( _v[1] > tri._v[1] )
        return false; 
    if ( _v[2] < tri._v[2] )
        return true; 

    return false; 
}

__host__ __device__ bool Triangle::operator == ( const Triangle& tri ) const
{
    return equal( tri );
}

__host__ __device__ bool Triangle::operator < ( const Triangle& tri ) const
{
    return lessThan( tri );
}

///////////////////////////////////////////////////////////////// Tetrahedron //

const int TetSegNum = 6;
const int TetSeg[ TetSegNum ][2] = {
    { 0, 1 },
    { 0, 2 },
    { 0, 3 },
    { 1, 2 },
    { 1, 3 },
    { 2, 3 },
};

void Tetrahedron::getSegments( Segment* segArr ) const
{
    for ( int i = 0; i < TetSegNum; ++i )
    {
        int vert[2] = { _v[ TetSeg[i][0] ], _v[ TetSeg[i][1] ] };

        if ( vert[0] > vert[1] ) std::swap( vert[0], vert[1] );

        const Segment seg   = { vert[0], vert[1] };
        segArr[i]           = seg;
    }

    return;
}

// Vertices of 4 triangles of tetra
const int TetTriNum     = 4;
const int TetTri[ TetTriNum ][3]  = {
    { 0, 1, 2 },
    { 0, 1, 3 },
    { 0, 2, 3 },
    { 1, 2, 3 },
};

void Tetrahedron::getTriangles( Triangle* triArr ) const
{
    for ( int i = 0; i < TetTriNum; ++i )
    {
        // Triangle vertices
        int vert[3] = { _v[ TetTri[i][0] ], _v[ TetTri[i][1] ], _v[ TetTri[i][2] ] };

        // Sort
        if ( vert[0] > vert[1] ) std::swap( vert[0], vert[1] );
        if ( vert[1] > vert[2] ) std::swap( vert[1], vert[2] );
        if ( vert[0] > vert[1] ) std::swap( vert[0], vert[1] );

        // Add triangle
        const Triangle tri  = { vert[0], vert[1], vert[2] };
        triArr[ i ]         = tri;
    }

    return;
}

void TetraMesh::setPoints( const Point3HVec& pointVec )
{
    _pointVec = pointVec;
    return;
}

void TetraMesh::setTetra( const TetraHVec& tetraVec )
{
    _tetraVec = tetraVec;
    return;
}

void TetraMesh::check()
{
    _checkEuler();

    exactinit();
    _checkOrientation();
    _checkInSphere();

    return;
}

void TetraMesh::_checkEuler() const
{
    const int v     = _getVertexCount();
    const int e     = _getSegmentCount();
    const int f     = _getTriangleCount();
    const int t     = ( int ) _tetraVec.size();
    const int euler = v - e + f - t;

    cout << "Euler Characteristic:" << endl;
    cout << "V: " << v << " E: " << e << " F: " << f << " T: " << t << " Euler: " << euler << endl;

    if ( 1 != euler )
    {
        cout << "Euler check failed!" << endl;
    }

    return;
}

int TetraMesh::_getVertexCount() const
{
    // Estimate space
    const int tetNum        = ( int ) _tetraVec.size();
    const int estPointNum   = tetNum * 4;

    // Reserve space
    IntVec vertVec;
    vertVec.reserve( estPointNum );

    // Add vertices
    for ( int ti = 0; ti < tetNum; ++ti )
    {
        const Tetrahedron& tet = _tetraVec[ ti ];

        for ( int vi = 0; vi < 4; ++vi )
            vertVec.push_back( tet._v[ vi ] );
    }

    // Sort and remove dups

    std::sort( vertVec.begin(), vertVec.end() );
    vertVec.erase( std::unique( vertVec.begin(), vertVec.end() ), vertVec.end() );

    const int vertNum = ( int ) vertVec.size();

    return vertNum;
}

int TetraMesh::_getSegmentCount() const
{
    // Estimate size
    const int tetNum    = ( int ) _tetraVec.size();
    const int estSegNum = ( int ) ( tetNum * TetSegNum );

    // Reserve space
    SegmentHVec segVec;
    segVec.reserve( estSegNum );

    // Read segments
    Segment segArr[ TetSegNum ];
    for ( int ti = 0; ti < tetNum; ++ti )
    {
        const Tetrahedron& tet = _tetraVec[ ti ];

        tet.getSegments( segArr );

        std::copy( segArr, segArr + TetSegNum, std::back_inserter( segVec ) );
    }

    // Sort and remove dups
    std::sort( segVec.begin(), segVec.end() );
    segVec.erase( std::unique( segVec.begin(), segVec.end() ), segVec.end() );

    const int segNum = ( int ) segVec.size();

    return segNum;
}

int TetraMesh::_getTriangleCount() const
{
    // Estimate size
    const int tetNum    = ( int ) _tetraVec.size();
    const int estTriNum = ( int ) ( tetNum * TetTriNum );

    // Reserve space
    TriangleHVec triVec;
    triVec.reserve( estTriNum );

    // Read triangles
    Triangle triArr[ TetTriNum ];
    for ( int ti = 0; ti < tetNum; ++ti )
    {
        const Tetrahedron& tet = _tetraVec[ ti ];

        tet.getTriangles( triArr );

        std::copy( triArr, triArr + TetTriNum, std::back_inserter( triVec ) );
    }

    // Sort and remove dups
    std::sort( triVec.begin(), triVec.end() );
    triVec.erase( std::unique( triVec.begin(), triVec.end() ), triVec.end() );

    const int triNum = ( int ) triVec.size();

    return triNum;
}

void TetraMesh::_checkOrientation()
{
    const int tetNum    = ( int ) _tetraVec.size();
    int failCount       = 0;

    for ( int ti = 0; ti < tetNum; ++ti )
    {
        const Tetrahedron& tet  = _tetraVec[ ti ];
        Point3* tp[4]           = { &_pointVec[ tet._v[0] ], &_pointVec[ tet._v[1] ], &_pointVec[ tet._v[2] ], &_pointVec[ tet._v[3] ] };
        const RealType ord      = orient3d( tp[0]->_p, tp[1]->_p, tp[2]->_p, tp[3]->_p );
        if ( ord > 0 )
            ++failCount;
    }

    if ( failCount > 0 )
    {
        cout << "Orientation check failed!!!" << endl;
        cout << "Tetra failures: " << failCount << endl;
    }
    else
    {
        cout << "Tetra orientation is correct!" << endl;
    }

    return;
}

void TetraMesh::_checkInSphere()
{
    const int tetNum    = ( int ) _tetraVec.size();
    int failCount       = 0;

    for ( int tetIdx = 0; tetIdx < tetNum; ++tetIdx )
    {
        const Tetrahedron& tet = _tetraVec[ tetIdx ];

        // Iterate 4 faces of tetra
        for ( int vi = 0; vi < 4; ++vi )
        {
            const int oppTetIdx = tet._opp[ vi ];

            if ( -1 == oppTetIdx )
                continue;

            assert( ( oppTetIdx >= 0 ) && ( oppTetIdx < tetNum ) && "Invalid opposite tetra index!" );

            // Check each pair only once
            if ( tetIdx > oppTetIdx )
                continue;

            // Check in-sphere

            const Tetrahedron& oppTet   = _tetraVec[ oppTetIdx ];
            const int oppVi             = oppTet.indexOfOpp( tetIdx );
            const int oppVert           = oppTet._v[ oppVi ];

            Point3* tp[5]       = { &_pointVec[ tet._v[0] ], &_pointVec[ tet._v[1] ], &_pointVec[ tet._v[2] ], &_pointVec[ tet._v[3] ], &_pointVec[ oppVert ] };
            const RealType side = insphere( tp[0]->_p, tp[1]->_p, tp[2]->_p, tp[3]->_p, tp[4]->_p );
            if ( side < 0 )
                ++failCount;
        }
    }

    if ( failCount > 0 )
    {
        cout << "In-sphere check failed!!!" << endl;
        cout << "Tetra failures: " << failCount << endl;
    }
    else
    {
        cout << "Tetra in-sphere is correct!" << endl;
    }

    return;
}

// Write out mesh as PLY file
void TetraMesh::writeToFile( const string& outFilename )
{
    ofstream outFile( outFilename.c_str() );

    if ( !outFile )
    {
        cerr << "Error opening output file: " << outFilename << "!" << endl;
        exit( 1 );
    }

    ////
    // Header
    ////

    const int pointNum = _pointVec.size();
    const int tetraNum = _tetraVec.size();

    outFile << "ply" << endl;
    outFile << "format ascii 1.0" << endl;
    outFile << "element vertex " << pointNum << endl;
    outFile << "property float x" << endl;
    outFile << "property float y" << endl;
    outFile << "property float z" << endl;
    outFile << "element face " << tetraNum * 3 << endl;
    outFile << "property list uchar int vertex_index" << endl;
    outFile << "end_header" << endl;

    ////
    // Points
    ////

    for ( int pi = 0; pi < pointNum; ++pi )
    {
        const Point3& pt = _pointVec[ pi ];

        for ( int vi = 0; vi < 3; ++vi )
        {
            outFile << pt._p[ vi ] << " ";
        }

        outFile << endl;
    }

    ////
    // Tetrahedron faces
    ////

    const int Faces[3][3] = {
        { 0, 1, 2 },
        { 0, 1, 3 },
        { 0, 2, 3 } };

    for ( int ti = 0; ti < tetraNum; ++ti )
    {
        const Tetrahedron& tet = _tetraVec[ ti ];

        for ( int fi = 0; fi < 3; ++fi )
        {
            outFile << "3 ";

            for ( int vi = 0; vi < 3; ++vi )
            {
                outFile << tet._v[ Faces[ fi ][ vi ] ] << " ";
            }

            outFile << endl;
        }
    }
    
    return;
}

////////////////////////////////////////////////////////////////////////////////
