/*
Author: Ashwin Nanjappa
Filename: GDelData.h

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
//                            GDelaunay Global Data
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "GDelCommon.h"

/////////////////////////////////////////////////////////////////////// Types //

struct PointData
{
    Point3DVec* _pointVec;
    Point3DVec* _scaledVec;
    int         _bitsPerIndex;

    void init( Point3HVec& pointHVec, Point3HVec& scaledHVec );
    void deinit();
    KerPointData toKernel();
};

struct TriangleData
{
    TriDVec*        _triVec[2];         // Triangles of stars
    TriOppDVec*     _triOppVec[2];
    IntDVec*        _triStarVec[2];     // Star of triangle
    TriStatusDVec*  _triStatusVec[2];

    void init();
    void deinit();
    void resize( int newSize, int arrId, const TriangleStatus& triStatus );
    int size( int vecId ) const;
    int totalSize() const;
};

struct StarData
{
    int                 _starNum;       // Number of stars
    TriangleData        _triData; 
    IntDVec*            _starTriMap[2]; // Index into triangle array for each star
    IntDVec*            _pointNumVec;   // Number of points in each star
    IntDVec*            _maxSizeVec;    // The farthest slot ever used by each star
    IntDVec*            _insCountVec;   // Number of insertions of each star

    void init( int pointNum );
    void deInit();
    KerStarData toKernel();

    // Expands input vector to hold old data
    template< typename T >
    void expandData( int oldSize, int newSize, IntDVec& oldNewMap, DeviceContainer< T >& inVec );
    void expandTriangles( int newSize, IntDVec& newTriMap );
};

struct MissingData
{
    IntDVec*    _memberVec; // All points
    IntDVec*    _leaderVec; // Leader of each point

    void init();
    void deInit();
    KerMissingData toKernel();
};

struct InsertionData
{
    IntDVec*    _vertVec;
    IntDVec*    _vertStarVec;
    IntDVec*    _starVertMap;
    IntDVec*    _shelveStarVec;
    IntDVec*    _shelveVertVec;

    void init();
    void deInit();
    KerInsertData toKernel();
};

struct HistoryData
{
    IntDVec*    _vertVec[2];        // All vertices inserted into star
    IntDVec*    _vertStarVec[2];    // Star of each vertex
    IntDVec*    _starVertMap[2];    // Map for above array

    void init();
    void deInit();
    KerHistoryData toKernel();
};

struct BeneathData
{
    TriPositionDVec*    _beneathTriPosVec;  // TriPos of a beneath triangle for each star
    TriPositionDVec*    _exactTriPosVec;
    IntDVec*            _flagVec;

    void init( int pointNum );
    void deInit();
    KerBeneathData toKernel();
};

struct ActiveData
{
    IntDVec*    _starVec;
    IntDVec*    _starTriMap;

    void init();
    void deInit();
    KerActiveData toKernel();
};

struct TetraData
{
    TetraDVec* _vec;

    void init();
    void deInit();
    KerTetraData toKernel();
};

////////////////////////////////////////////////////////////////////////////////
