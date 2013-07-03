/*
Author: Ashwin Nanjappa
Filename: Pba.cu

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
//                             PBA Main Code
////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////// Headers //

// Self
#include "Pba.h"
#include "pba3D.h"

// Project
#include "Geometry.h"

// External
#include <iostream>

using namespace std;

///////////////////////////////////////////////////////////////////// Globals //

int fboSize     = 64;
int nVertices   = 100; 

int phase1Band  = 1; 
int phase2Band  = 1; 
int phase3Band  = 2; 

int *buffer[2]; 
int *dInputVoronoi;
int *dOutputVoronoi;
int *dPointIndexGrid;

int gridNum;

Point3DVec* DPointVec;

////////////////////////////////////////////////////////////////////////////////

#define TOID(x, y, z, w)    ((z) * (w) * (w) + (y) * (w) + (x))

/**********************************************************************************************
 * Initialization
 **********************************************************************************************/
void initialization()
{
    pba3DInitialization(fboSize); 

    gridNum = fboSize * fboSize * fboSize;

    // Allocate 2 buffers
    cudaMalloc( ( void** ) &buffer[0], gridNum * sizeof( int ) ); 
    cudaMalloc( ( void** ) &buffer[1], gridNum * sizeof( int ) ); 

    dInputVoronoi   = buffer[0]; 
    dOutputVoronoi  = buffer[1]; 

    return;
}

void pbaInit( int volWidth, Point3DVec* inPointDVec )
{
    // Set parameters
    fboSize   = volWidth;
    DPointVec = inPointDVec;
    nVertices = DPointVec->size();

    initialization();
    setPointsInGrid( *DPointVec, dInputVoronoi );

    return;
}

void doPba( int** dOutGrid )
{
    pba3DVoronoiDiagram( dInputVoronoi, &dOutputVoronoi, phase1Band, phase2Band, phase3Band );

    // One of the two buffers is the output, use the other one to store the indices
    if ( buffer[0] == dOutputVoronoi ) 
        dPointIndexGrid = buffer[1]; 
    else
        dPointIndexGrid = buffer[0]; 

    setPointIndicesInGrid( *DPointVec, dPointIndexGrid ); 
    setIndexInGrid( fboSize, dPointIndexGrid, dOutputVoronoi );

    // Return grid
    *dOutGrid = dOutputVoronoi;

    return;
}

void pbaDeinit()
{
    pba3DDeinitialization();
    return;
}
