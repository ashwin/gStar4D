/*
Author: Cao Thanh Tung
Filename: pba3DHost.cu

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

#include <device_functions.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <fstream>
using namespace std;

#include "pba3D.h"
#include "Geometry.h"
#include "CudaWrapper.h"

// Parameters for CUDA kernel executions
#define BLOCKX      32
#define BLOCKY      4
#define BLOCKXY     16

#define PBA_INFINITY    0x3ff

/****** Global Variables *******/
int **pbaTextures;   

int pbaMemSize; 
int pbaCurrentBuffer; 
int pbaTexSize;  

texture<int> pbaTexColor; 
texture<int> pbaTexLinks; 
texture<short> pbaTexPointer; 

/********* Kernels ********/
#include "pba3DKernel.h"

///////////////////////////////////////////////////////////////////////////
//
// Initialize necessary memory for 3D Voronoi Diagram computation
// - textureSize: The size of the Discrete Voronoi Diagram (width = height)
//
///////////////////////////////////////////////////////////////////////////
void pba3DInitialization(int fboSize)
{
    pbaTexSize  = fboSize; 
    pbaTextures = (int **) malloc(2 * sizeof(int *)); 
    pbaMemSize  = pbaTexSize * pbaTexSize * pbaTexSize * sizeof(int); 

    // Allocate 2 textures
    //cudaMalloc((void **) &pbaTextures[0], pbaMemSize); 
    //cudaMalloc((void **) &pbaTextures[1], pbaMemSize); 
}

///////////////////////////////////////////////////////////////////////////
//
// Deallocate all allocated memory
//
///////////////////////////////////////////////////////////////////////////
void pba3DDeinitialization()
{
    free(pbaTextures);

    return;
}

// Copy input to GPU 
void pba3DInitializeInput(int *input, int *output)
{
    //cudaMemcpy(pbaTextures[0], input, pbaMemSize, cudaMemcpyHostToDevice); 
    pbaTextures[0] = input; 
    pbaTextures[1] = output; 

    // Set Current Source Buffer
    pbaCurrentBuffer = 0;
}

// In-place transpose a cubic texture. 
// Transposition are performed on each XY plane. 
// Point coordinates are also swapped. 
void pba3DTransposeXY(int *texture)
{
    dim3 block(BLOCKXY, BLOCKXY); 
    dim3 grid((pbaTexSize / BLOCKXY) * pbaTexSize, pbaTexSize / BLOCKXY); 

    kernelTransposeXY<<< grid, block >>>(texture, pbaTexSize); 
    CudaCheckError();
}

// Phase 1 of PBA. m1 must divides texture size
// Sweeping are done along the Z axiz. 
void pba3DColorZAxis(int m1) 
{
    dim3 block = dim3(BLOCKX, BLOCKY); 
    dim3 grid = dim3((pbaTexSize / block.x) * m1, pbaTexSize / block.y); 

    CudaSafeCall( cudaBindTexture(0, pbaTexColor, pbaTextures[pbaCurrentBuffer]) ); 

    kernelFloodZ<<< grid, block >>>(pbaTextures[1 - pbaCurrentBuffer], pbaTexSize, pbaTexSize / block.x, pbaTexSize / m1); 
    CudaCheckError();

    pbaCurrentBuffer = 1 - pbaCurrentBuffer; 

    if (m1 > 1)
    {
        // Passing information between bands
        CudaSafeCall( cudaBindTexture(0, pbaTexColor, pbaTextures[pbaCurrentBuffer]) ); 

        kernelPropagateInterband<<< grid, block >>>(pbaTextures[1 - pbaCurrentBuffer], pbaTexSize, pbaTexSize / block.x, pbaTexSize / m1); 
        CudaCheckError();

        CudaSafeCall( cudaBindTexture(0, pbaTexLinks, pbaTextures[1 - pbaCurrentBuffer]) ); 

        kernelUpdateVertical<<< grid, block >>>(pbaTextures[pbaCurrentBuffer], pbaTexSize, pbaTexSize / block.x, pbaTexSize / m1); 
        CudaCheckError();
    }
}

// Phase 2 of PBA. m2 must divides texture size. 
// This method work along the Y axis
void pba3DComputeProximatePointsYAxis(int m2) 
{
    int iStack = 1 - pbaCurrentBuffer; 
    int iForward = pbaCurrentBuffer; 

    dim3 block = dim3(BLOCKX, BLOCKY); 
    dim3 grid = dim3((pbaTexSize / block.x) * m2, pbaTexSize / block.y); 

    // Compute proximate points locally in each band
    CudaSafeCall( cudaBindTexture(0, pbaTexColor, pbaTextures[pbaCurrentBuffer]) ); 

    kernelMaurerAxis<<< grid, block >>>(pbaTextures[iStack], pbaTexSize, pbaTexSize / block.x, pbaTexSize / m2); 
    CudaCheckError();

    // Construct forward pointers
    CudaSafeCall( cudaBindTexture(0, pbaTexLinks, pbaTextures[iStack]) ); 

    kernelCreateForwardPointers<<< grid, block >>>((short *) pbaTextures[iForward], pbaTexSize, pbaTexSize / block.x, pbaTexSize / m2); 
    CudaCheckError();

    CudaSafeCall( cudaBindTexture(0, pbaTexPointer, pbaTextures[iForward], pbaTexSize * pbaTexSize * pbaTexSize * sizeof( short ) ) ); 

    // Repeatly merging two bands into one
    for (int noBand = m2; noBand > 1; noBand /= 2)
    {
        grid = dim3((pbaTexSize / block.x) * (noBand / 2), pbaTexSize / block.y); 
        kernelMergeBands<<< grid, block >>>(pbaTextures[iStack], 
            (short *) pbaTextures[iForward], pbaTexSize, pbaTexSize / block.x, pbaTexSize / noBand); 
        CudaCheckError();
    }

    CudaSafeCall( cudaUnbindTexture(pbaTexLinks) ); 
    CudaSafeCall( cudaUnbindTexture(pbaTexColor) ); 
    CudaSafeCall( cudaUnbindTexture(pbaTexPointer) ); 
}

// Phase 3 of PBA. m3 must divides texture size
// This method color along the Y axis
void pba3DColorYAxis(int m3) 
{
    dim3 block = dim3(BLOCKX, m3); 
    dim3 grid = dim3(pbaTexSize / block.x, pbaTexSize); 

    CudaSafeCall( cudaBindTexture(0, pbaTexColor, pbaTextures[1 - pbaCurrentBuffer] ) ); 

    kernelColorAxis<<< grid, block >>>(pbaTextures[pbaCurrentBuffer], pbaTexSize); 
    CudaCheckError();

    CudaSafeCall( cudaUnbindTexture(pbaTexColor) ); 

    return;
}

void pba3DCompute(int m1, int m2, int m3)
{
    /************* Compute Z axis *************/
    // --> (X, Y, Z)
    pba3DColorZAxis(m1); 

    /************* Compute Y axis *************/
    // --> (X, Y, Z)
    pba3DComputeProximatePointsYAxis(m2);
    pba3DColorYAxis(m3); 

    // --> (Y, X, Z)
    pba3DTransposeXY(pbaTextures[pbaCurrentBuffer]); 

    /************** Compute X axis *************/
    // Compute X axis
    pba3DComputeProximatePointsYAxis(m2);
    pba3DColorYAxis(m3); 

    // --> (X, Y, Z)
    pba3DTransposeXY(pbaTextures[pbaCurrentBuffer]); 
}

// Compute 3D Voronoi diagram
// Input: a 3D texture. Each pixel is an integer encoding 3 coordinates. 
//    For each site at (x, y, z), the pixel at coordinate (x, y, z) should contain 
//    the encoded coordinate (x, y, z). Pixels that are not sites should contain 
//    the integer MARKER. Use ENCODE (and DECODE) macro to encode (and decode).
// See original paper for the effect of the three parameters: 
//     phase1Band, phase2Band, phase3Band
// Parameters must divide textureSize
// Note: input texture will be released after this.
void pba3DVoronoiDiagram(int *dInput, int **dOutput, 
                         int phase1Band, int phase2Band, int phase3Band) 
{
    // Initialization
    pba3DInitializeInput(dInput, *dOutput); 
    
    // Compute the 3D Voronoi Diagram
    pba3DCompute(phase1Band, phase2Band, phase3Band); 

    // Pass back the result
    *dOutput = pbaTextures[pbaCurrentBuffer];
    
    return;
}

// A function to draw points onto GPU texture
void setPointsInGrid( Point3DVec& pointDVec, int *dInputVoronoi )
{   
    const int BlocksPerGrid     = 64; 
    const int ThreadsPerBlock   = 256; 

    CudaSafeCall( cudaMemset( dInputVoronoi, MARKER, pbaMemSize ) ); 

    kerSetPointsInGrid<<< BlocksPerGrid, ThreadsPerBlock >>>(
        thrust::raw_pointer_cast( &pointDVec[0] ),
        ( int ) pointDVec.size(),
        dInputVoronoi,
        pbaTexSize
    ); 
    CudaCheckError();

    return;
}

// A function to draw point's IDs onto GPU texture
void setPointIndicesInGrid( Point3DVec& pointDVec, int* dMapToID )
{   
    const int BlocksPerGrid     = 64; 
    const int ThreadsPerBlock   = 256; 

    kerSetPointIndicesInGrid<<< BlocksPerGrid, ThreadsPerBlock >>>(
        thrust::raw_pointer_cast( &pointDVec[0] ),
        ( int ) pointDVec.size(),
        dMapToID,
        pbaTexSize
    ); 
    CudaCheckError();

    return;
}

void setIndexInGrid( int gridWidth, int* dPointIndexGrid, int* dGrid )
{
    const int BlocksPerGrid     = 64; 
    const int ThreadsPerBlock   = 256; 

    kerSetIndexInGrid<<< BlocksPerGrid, ThreadsPerBlock >>>( gridWidth, dPointIndexGrid, dGrid );
    CudaCheckError();

    // Free grid
    CudaSafeCall( cudaFree( dPointIndexGrid ) );

    return;
}

