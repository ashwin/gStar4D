/*
Author: Ashwin Nanjappa
Filename: GDelPredDevice.h

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
//                           Predicates for CUDA
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "GDelShewchukDevice.h"

////////////////////////////////////////////////////////////////////////////////

__device__ Orient detToOrient( RealType det )
{
    return ( det > 0 ) ? OrientPos : ( ( det < 0 ) ? OrientNeg : OrientZero );
}

__device__ Orient flipOrient( Orient ord )
{
    //CudaAssert( ( OrientZero != ord ) && "Zero orientation cannot be flipped!" );
    if ( OrientZero == ord) 
        return ord; 
    return ( OrientPos == ord ) ? OrientNeg : OrientPos;
}

__device__ Orient shewchukOrient3D
(
const RealType* predConsts,
const RealType* p0,
const RealType* p1,
const RealType* p2,
const RealType* p3
)
{
    RealType det = orient3dfast( predConsts, p0, p1, p2, p3 ); 

    // Need exact check
    if ( det == 0.0 )
    {
        det = orient3dexact( predConsts, p0, p1, p2, p3 );
    }

    return detToOrient( det );
}

__device__ Orient orient1DLifted
(
const RealType* predConsts,
const RealType* p0,
const RealType* p1,
bool            lifted
)
{
    const RealType det = orient1dexact_lifted( predConsts, p0, p1, lifted );
    return detToOrient( det );
}

__device__ Orient orient2DLifted
(
const PredicateInfo predInfo,
const RealType*     p0,
const RealType*     p1,
const RealType*     p2,
bool                lifted
)
{
    const RealType det = orient2dexact_lifted( predInfo, p0, p1, p2, lifted );
    return detToOrient( det );
}

__device__ Orient orient3DLifted
(
const PredicateInfo predInfo,
const RealType*     p0,
const RealType*     p1,
const RealType*     p2,
const RealType*     p3,
bool                lifted
)
{
    const RealType det = orient3dexact_lifted( predInfo, p0, p1, p2, p3, lifted );
    return detToOrient( det );
}

__forceinline__ __device__ Orient orientation4Fast
(
PredicateInfo   predInfo,
KerPointData    pointData,
int             pi0,
int             pi1,
int             pi2,
int             pi3,
int             pi4
)
{
    ////
    // Use insphere3D as orient4D
    ////

    const Point3* ptArr = pointData._pointArr;
    const Point3* pt[]  = { &( ptArr[ pi0 ] ), &( ptArr[ pi1 ] ), &( ptArr[ pi2 ] ), &( ptArr[ pi3 ] ), &( ptArr[ pi4 ] ) };
    const RealType det  = insphere( predInfo, pt[0]->_p, pt[1]->_p, pt[2]->_p, pt[3]->_p, pt[4]->_p );
    const Orient ord    = detToOrient( det );
    return ord;
}

//////
// Note: Only called when simple orientation4 returns ZERO
// Reference: Simulation of Simplicity paper by Edelsbrunner and Mucke
//////

__device__ Orient _orientation4SoS
(
const PredicateInfo predInfo,
const RealType*     p0,
const RealType*     p1,
const RealType*     p2,
const RealType*     p3,
const RealType*     p4,
int                 pi0,
int                 pi1,
int                 pi2,
int                 pi3,
int                 pi4
)
{
    const int DIM = 4;
    const int NUM = DIM + 1;

    // Sort indices & note their order

    int idx[NUM]    = { pi0, pi1, pi2, pi3, pi4 };
    int ord[NUM]    = { 0, 1, 2, 3, 4 };
    int swapCount   = 0;

    for ( int i = 0; i < ( NUM - 1 ); ++i )
    {
        for ( int j = ( NUM - 1 ); j > i; --j )
        {
            if ( idx[j] < idx[j - 1] )
            {
                cuSwap( idx[j], idx[j - 1] );
                cuSwap( ord[j], ord[j - 1] );   // Note order
                ++swapCount;
            }
        }
    }

    // Sort points in sorted index order.

    const RealType* pt4Arr[NUM] = { p0, p1, p2, p3, p4 };
    const RealType* ip[NUM];

    for ( int i = 0; i < NUM; ++i )
        ip[i] = pt4Arr[ ord[i] ];

    // Calculate determinants

    RealType op[NUM-1][DIM-1]   = {0};
    Orient orient           = OrientZero;
    int depth               = 0;

    // Setup determinants
    while ( OrientZero == orient )
    {
        ++depth;    // Increment depth, begins from 1

        switch ( depth )
        {
        case 0:
            break;

        case 1:
            op[0][0] = ip[1][0];    op[0][1] = ip[1][1];    op[0][2] = ip[1][2];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];    op[1][2] = ip[2][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 2:
            op[0][0] = ip[1][0];    op[0][1] = ip[1][1];    op[0][2] = ip[1][2];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];    op[1][2] = ip[2][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 3:
            op[0][0] = ip[1][0];    op[0][1] = ip[1][2];    op[0][2] = ip[1][1];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][2];    op[1][2] = ip[2][1];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][2];    op[2][2] = ip[3][1];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][2];    op[3][2] = ip[4][1];
            break;

        case 4:
            op[0][0] = ip[1][1];    op[0][1] = ip[1][2];    op[0][2] = ip[1][0];
            op[1][0] = ip[2][1];    op[1][1] = ip[2][2];    op[1][2] = ip[2][0];
            op[2][0] = ip[3][1];    op[2][1] = ip[3][2];    op[2][2] = ip[3][0];
            op[3][0] = ip[4][1];    op[3][1] = ip[4][2];    op[3][2] = ip[4][0];
            break;

        case 5:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];    op[1][2] = ip[2][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 6:
            op[0][0] = ip[2][0];    op[0][1] = ip[2][1];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 7:
            op[0][0] = ip[2][0];    op[0][1] = ip[2][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][2];
            break;

        case 8:
            op[0][0] = ip[2][1];    op[0][1] = ip[2][2];
            op[1][0] = ip[3][1];    op[1][1] = ip[3][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][2];
            break;

        case 9:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];    op[1][2] = ip[2][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 10:
            op[0][0] = ip[2][0];    op[0][1] = ip[2][1];    op[0][2] = ip[2][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];    op[2][2] = ip[4][2];
            break;

        case 11:
            op[0][0] = ip[2][1];    op[0][1] = ip[2][0];    op[0][2] = ip[2][2];
            op[1][0] = ip[3][1];    op[1][1] = ip[3][0];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][0];    op[2][2] = ip[4][2];
            break;
            
        case 12:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][2];    op[0][2] = ip[0][1];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][2];    op[1][2] = ip[2][1];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][2];    op[2][2] = ip[3][1];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][2];    op[3][2] = ip[4][1];
            break;

        case 13:
            op[0][0] = ip[2][2];    op[0][1] = ip[2][0];    op[0][2] = ip[2][1];
            op[1][0] = ip[3][2];    op[1][1] = ip[3][0];    op[1][2] = ip[3][1];
            op[2][0] = ip[4][2];    op[2][1] = ip[4][0];    op[2][2] = ip[4][1];
            break;

        case 14:
            op[0][0] = ip[0][1];    op[0][1] = ip[0][2];    op[0][2] = ip[0][0];
            op[1][0] = ip[2][1];    op[1][1] = ip[2][2];    op[1][2] = ip[2][0];
            op[2][0] = ip[3][1];    op[2][1] = ip[3][2];    op[2][2] = ip[3][0];
            op[3][0] = ip[4][1];    op[3][1] = ip[4][2];    op[3][2] = ip[4][0];
            break;

        case 15:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][1];    op[1][2] = ip[1][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 16:
            op[0][0] = ip[1][0];    op[0][1] = ip[1][1];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 17:
            op[0][0] = ip[1][0];    op[0][1] = ip[1][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][2];
            break;

        case 18:
            op[0][0] = ip[1][1];    op[0][1] = ip[1][2];
            op[1][0] = ip[3][1];    op[1][1] = ip[3][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][2];
            break;

        case 19:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 20:
            op[0][0] = ip[3][0];
            op[1][0] = ip[4][0];
            break;

        case 21:
            op[0][0] = ip[3][1];
            op[1][0] = ip[4][1];
            break;

        case 22:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][2];
            break;

        case 23:
            op[0][0] = ip[3][2];
            op[1][0] = ip[4][2];
            break;

        case 24:
            op[0][0] = ip[0][1];    op[0][1] = ip[0][2];
            op[1][0] = ip[3][1];    op[1][1] = ip[3][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][2];
            break;

        case 25:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][1];    op[1][2] = ip[1][2];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][1];    op[2][2] = ip[3][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 26:
            op[0][0] = ip[1][0];    op[0][1] = ip[1][1];    op[0][2] = ip[1][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];    op[2][2] = ip[4][2];
            break;

        case 27:
            op[0][0] = ip[1][1];    op[0][1] = ip[1][0];    op[0][2] = ip[1][2];
            op[1][0] = ip[3][1];    op[1][1] = ip[3][0];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][0];    op[2][2] = ip[4][2];
            break;

        case 28:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[3][0];    op[1][1] = ip[3][1];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];    op[2][2] = ip[4][2];
            break;

        case 29:
            op[0][0] = ip[3][0];    op[0][1] = ip[3][1];    op[0][2] = ip[3][2];
            op[1][0] = ip[4][0];    op[1][1] = ip[4][1];    op[1][2] = ip[4][2];
            break;

        case 30:
            op[0][0] = ip[0][1];    op[0][1] = ip[0][0];    op[0][2] = ip[0][2];
            op[1][0] = ip[3][1];    op[1][1] = ip[3][0];    op[1][2] = ip[3][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][0];    op[2][2] = ip[4][2];
            break;
            
        case 31:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][2];    op[0][2] = ip[0][1];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][2];    op[1][2] = ip[1][1];
            op[2][0] = ip[3][0];    op[2][1] = ip[3][2];    op[2][2] = ip[3][1];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][2];    op[3][2] = ip[4][1];
            break;

        case 32:
            op[0][0] = ip[1][2];    op[0][1] = ip[1][0];    op[0][2] = ip[1][1];
            op[1][0] = ip[3][2];    op[1][1] = ip[3][0];    op[1][2] = ip[3][1];
            op[2][0] = ip[4][2];    op[2][1] = ip[4][0];    op[2][2] = ip[4][1];
            break;

        case 33:
            op[0][0] = ip[0][2];    op[0][1] = ip[0][0];    op[0][2] = ip[0][1];
            op[1][0] = ip[3][2];    op[1][1] = ip[3][0];    op[1][2] = ip[3][1];
            op[2][0] = ip[4][2];    op[2][1] = ip[4][0];    op[2][2] = ip[4][1];
            break;

        case 34:
            op[0][0] = ip[0][1];    op[0][1] = ip[0][2];    op[0][2] = ip[0][0];
            op[1][0] = ip[1][1];    op[1][1] = ip[1][2];    op[1][2] = ip[1][0];
            op[2][0] = ip[3][1];    op[2][1] = ip[3][2];    op[2][2] = ip[3][0];
            op[3][0] = ip[4][1];    op[3][1] = ip[4][2];    op[3][2] = ip[4][0];
            break;

        case 35:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];    op[0][2] = ip[0][2];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][1];    op[1][2] = ip[1][2];
            op[2][0] = ip[2][0];    op[2][1] = ip[2][1];    op[2][2] = ip[2][2];
            op[3][0] = ip[4][0];    op[3][1] = ip[4][1];    op[3][2] = ip[4][2];
            break;

        case 36:
            op[0][0] = ip[1][0];    op[0][1] = ip[1][1];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 37:
            op[0][0] = ip[1][0];    op[0][1] = ip[1][2];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][2];
            break;

        case 38:
            op[0][0] = ip[1][1];    op[0][1] = ip[1][2];
            op[1][0] = ip[2][1];    op[1][1] = ip[2][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][2];
            break;

        case 39:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 40:
            op[0][0] = ip[2][0];
            op[1][0] = ip[4][0];
            break;

        case 41:
            op[0][0] = ip[2][1];
            op[1][0] = ip[4][1];
            break;

        case 42:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][2];
            op[1][0] = ip[2][0];    op[1][1] = ip[2][2];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][2];
            break;

        case 43:
            op[0][0] = ip[2][2];
            op[1][0] = ip[4][2];
            break;

        case 44:
            op[0][0] = ip[0][1];    op[0][1] = ip[0][2];
            op[1][0] = ip[2][1];    op[1][1] = ip[2][2];
            op[2][0] = ip[4][1];    op[2][1] = ip[4][2];
            break;

        case 45:
            op[0][0] = ip[0][0];    op[0][1] = ip[0][1];
            op[1][0] = ip[1][0];    op[1][1] = ip[1][1];
            op[2][0] = ip[4][0];    op[2][1] = ip[4][1];
            break;

        case 46:
            op[0][0] = ip[1][0];
            op[1][0] = ip[4][0];
            break;

        case 47:
            op[0][0] = ip[1][1];
            op[1][0] = ip[4][1];
            break;

        case 48:
            op[0][0] = ip[0][0];
            op[1][0] = ip[4][0];
            break;

        case 49:
            // See below for result
            break;

        default:
            CudaAssert( false && "Invalid SoS depth!" );
            break;

        }   // switch ( depth )

        // *** Calculate determinant

        bool lifted = false; 

        switch ( depth )
        {
        // 3D orientation involving the lifted coordinate
        case 2:
        case 3:
        case 4:
        case 9:
        case 12:
        case 14:
        case 25:
        case 31:
        case 34:

        // 2D orientation involving the lifted coordinate
        case 10:
        case 11:
        case 13:
        case 26:
        case 27:
        case 28:
        case 30:
        case 32:
        case 33:

        // 1D orientation involving the lifted coordinate
        case 29:
            {
                lifted = true; 
            }
            break;
        }

        switch ( depth )
        {
        // 3D orientation determinant
        case 1:
        case 5:
        case 15:
        case 35:

        // 3D orientation involving the lifted coordinate
        case 2:
        case 3:
        case 4:
        case 9:
        case 12:
        case 14:
        case 25:
        case 31:
        case 34:
            {
                orient = orient3DLifted( predInfo, op[0], op[1], op[2], op[3], lifted );
            }
            break;

        // 2D orientation determinant
        case 6:
        case 7:
        case 8:
        case 16:
        case 17:
        case 18:
        case 19:
        case 22:
        case 24:
        case 36:
        case 37:
        case 38:
        case 39:
        case 42:
        case 44:
        case 45:

        // 2D orientation involving the lifted coordinate
        case 10:
        case 11:
        case 13:
        case 26:
        case 27:
        case 28:
        case 30:
        case 32:
        case 33:
            {
                orient = orient2DLifted( predInfo, op[0], op[1], op[2], lifted );
            }
            break;

        // 1D orientation determinant
        case 20:
        case 21:
        case 23:
        case 40:
        case 41:
        case 43:
        case 46:
        case 47:
        case 48:

        // 1D orientation involving the lifted coordinate
        case 29:
            {
                orient = orient1DLifted( predInfo._consts, op[0], op[1], lifted );
            }
            break;

        case 49:
            // Last depth, always POS
            orient = OrientPos;
            break;

        default:
            CudaAssert( false && "Invalid SoS depth!" );
            break;
        }
    }   // while ( 0 == orient )

    ////
    // Flip result for certain depths. (See SoS paper.)
    ////

    switch ( depth )
    {
    // -ve result
    case 1:
    case 3:
    case 7:
    case 9:
    case 11:
    case 14:
    case 16:
    case 15:
    case 18:
    case 20:
    case 22:
    case 23:
    case 26:
    case 30:
    case 31:
    case 32:
    case 37:
    case 39:
    case 41:
    case 44:
    case 46:
        orient = flipOrient( orient );
        break;

    default:
        // Do nothing
        break;
    }

    ////
    // Flip result for odd swap count
    ////

    if ( ( swapCount % 2 ) != 0 )
        orient = flipOrient( orient );

    return orient;
}

__device__ Orient orientation4SoS
(
PredicateInfo   predInfo,
KerPointData    pointData,
int             pi0,
int             pi1,
int             pi2,
int             pi3,
int             pi4
)
{
    ////
    // Use insphere3D as orient4D
    ////

    const Point3* ptArr  = pointData._pointArr;
    const RealType* pt[] = { ptArr[ pi0 ]._p, ptArr[ pi1 ]._p, ptArr[ pi2 ]._p, ptArr[ pi3 ]._p, ptArr[ pi4 ]._p };
    const RealType det   = insphereexact( predInfo, pt[0], pt[1], pt[2], pt[3], pt[4]);
    const Orient ord     = detToOrient( det );

    if ( OrientZero != ord ) return ord;

    ////
    // SoS with 4D coordinates
    ////

    return _orientation4SoS( predInfo, pt[0], pt[1], pt[2], pt[3], pt[4], pi0, pi1, pi2, pi3, pi4 );
}

////////////////////////////////////////////////////////////////////////////////
