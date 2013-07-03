/*
Author: Cao Thanh Tung
Filename: pba3D.h

Copyright (c) 2010, School of Computing, National University of Singapore. 
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

#ifndef __CUDA_H__
#define __CUDA_H__

#include "Geometry.h"

// Initialize CUDA and allocate memory
// textureSize is 2^k with k >= 5
void pba3DInitialization(int textureSize); 

// Deallocate memory in GPU
void pba3DDeinitialization(); 

void setPointsInGrid( Point3DVec&, int * );
void setPointIndicesInGrid( Point3DVec&, int * );
void setIndexInGrid( int, int*, int* );

// Compute 3D Voronoi diagram
// Input: a 3D texture. Each pixel is an integer encoding 3 coordinates. 
//    For each site at (x, y, z), the pixel at coordinate (x, y, z) should contain 
//    the encoded coordinate (x, y, z). Pixels that are not sites should contain 
//    the integer MARKER. Use ENCODE (and DECODE) macro to encode (and decode).
// Output: a 3D texture. Each pixel is an integer encoding 3 coordinates 
//    of its nearest site. 
// See original paper for the effect of the three parameters: 
//     phase1Band, phase2Band, phase3Band
// Parameters must divide textureSize
void pba3DVoronoiDiagram( int *, int**, int , int , int );

#define MARKER      -1
#define MAX_INT     201326592

#define ENCODE(x, y, z)  (((x) << 20) | ((y) << 10) | (z))
#define DECODE(value, x, y, z) \
    x = (value) >> 20; \
    y = ((value) >> 10) & 0x3ff; \
    z = (value) & 0x3ff

#define GET_X(value)    ((value) >> 20)
#define GET_Y(value)    (((value) >> 10) & 0x3ff)
#define GET_Z(value)    (((value) == MARKER) ? MAX_INT : ((value) & 0x3ff))

#endif
