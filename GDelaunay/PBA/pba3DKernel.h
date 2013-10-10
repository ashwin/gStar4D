/*
Author: Cao Thanh Tung
Filename: pba3DKernel.h

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

#define TOID(x, y, z, w)    (__mul24(__mul24(z, w) + (y), w) + (x))

// Rotate (X, Y, Z) to (Y, X, Z)
#define ROTATEXY(x)   ((((x) & 0xffc00) << 10) | \
                       (((x) & 0x3ff00000) >> 10) | \
                       ((x) & 0x3ff))

__global__ void kernelTransposeXY(int *data, int width)
{
    __shared__ int block1[BLOCKXY][BLOCKXY + 1];
    __shared__ int block2[BLOCKXY][BLOCKXY + 1];

    int blkX = blockIdx.y; 
    int blkY = blockIdx.x / width; 
    int blkZ = blockIdx.x % width; 

    if (blkX > blkY) 
        return ; 
    
    int x, y, z, id1, id2; 
    int pixel; 
    
    blkX = __mul24(blkX, BLOCKXY); 
    blkY = __mul24(blkY, BLOCKXY); 
    z = blkZ * width; 

    // read the cube into shared memory
    x = blkX + threadIdx.x;
    y = blkY + threadIdx.y;
    id1 = ((z + y) * width) + x;
    block1[threadIdx.y][threadIdx.x] = data[id1];

    x = blkY + threadIdx.x;
    y = blkX + threadIdx.y;
    id2 = ((z + y) * width) + x;
    block2[threadIdx.y][threadIdx.x] = data[id2];

    __syncthreads();

    // write the rotated cube to global memory
    pixel = block1[threadIdx.x][threadIdx.y];
    data[id2] = ROTATEXY(pixel); 
    pixel = block2[threadIdx.x][threadIdx.y];
    data[id1] = ROTATEXY(pixel); 
}

// Flood along the Z axis
__global__ void kernelFloodZ(int *output, int size, int mod, int bandSize) 
{
    int blkX = blockIdx.x % mod; 
    int band = blockIdx.x / mod; 

    int tx = blkX * blockDim.x + threadIdx.x; 
    int ty = blockIdx.y * blockDim.y + threadIdx.y; 
    int tz = band * bandSize; 

    int plane = size * size; 
    int id = TOID(tx, ty, tz, size); 
    int pixel1, pixel2; 

    pixel1 = MARKER; 

    // Sweep down
    for (int i = 0; i < bandSize; i++, id += plane) {
        pixel2 = tex1Dfetch(pbaTexColor, id); 

        if (pixel2 != MARKER) 
            pixel1 = pixel2; 

        output[id] = pixel1; 
    }

    int dist1, dist2, nz; 

    id -= plane + plane; 

    // Sweep up
    for (int i = bandSize - 2; i >= 0; i--, id -= plane) {
        nz = GET_Z(pixel1); 
        dist1 = abs(nz - (tz + i)); 
        
        pixel2 = output[id];
        nz = GET_Z(pixel2); 
        dist2 = abs(nz - (tz + i)); 

        if (dist2 < dist1) 
            pixel1 = pixel2; 

        output[id] = pixel1; 
    }
}

__global__ void kernelPropagateInterband(int *output, int size, int mod, int bandSize) 
{
    int blkX = blockIdx.x % mod; 
    int band = blockIdx.x / mod; 

    int tx = blkX * blockDim.x + threadIdx.x; 
    int ty = blockIdx.y * blockDim.y + threadIdx.y; 

    int inc = bandSize * size * size; 
    int nz, nid, nDist, myDist; 
    int pixel; 

    // Top row, look backward
    int tz = __mul24(band, bandSize); 
    int topId = TOID(tx, ty, tz, size); 
    int bottomId = TOID(tx, ty, tz + bandSize - 1, size); 

    pixel = tex1Dfetch(pbaTexColor, topId); 
    nz = GET_Z(pixel); 
    myDist = abs(nz - tz); 

    for (nid = bottomId - inc; nid >= 0; nid -= inc) {
        pixel = tex1Dfetch(pbaTexColor, nid); 

        if (pixel != MARKER) { 
            nz = pixel & 0x3ff; 
            nDist = abs(nz - tz); 

            if (nDist < myDist) 
                output[topId] = pixel; 
                
            break;  
        }
    }

    // Last row, look downward
    tz = tz + bandSize - 1; 
    pixel = tex1Dfetch(pbaTexColor, bottomId);
    nz = GET_Z(pixel); 
    myDist = abs(nz - tz); 

    for (int ii = tz + 1, nid = topId + inc; ii < size; ii += bandSize, nid += inc) {
        pixel = tex1Dfetch(pbaTexColor, nid); 

        if (pixel != MARKER) { 
            nz = pixel & 0x3ff; 
            nDist = abs(nz - tz); 

            if (nDist < myDist) 
                output[bottomId] = pixel; 
                
            break; 
        }
    }
}

__global__ void kernelUpdateVertical(int *output, int size, int mod, int bandSize) 
{
    int blkX = blockIdx.x % mod; 
    int band = blockIdx.x / mod; 

    int tx = blkX * blockDim.x + threadIdx.x; 
    int ty = blockIdx.y * blockDim.y + threadIdx.y; 
    int tz = band * bandSize; 
    int id = TOID(tx, ty, tz, size); 
    int plane = size * size; 

    int top = tex1Dfetch(pbaTexLinks, id); 
    int bottom = tex1Dfetch(pbaTexLinks, TOID(tx, ty, tz + bandSize - 1, size)); 
    int topZ = GET_Z(top); 
    int bottomZ = GET_Z(bottom); 
    int pixel; 

    int dist, myDist, nz; 

    for (int i = 0; i < bandSize; i++, id += plane) {
        pixel = tex1Dfetch(pbaTexColor, id); 
        nz = GET_Z(pixel); 
        myDist = abs(nz - (tz + i)); 

        dist = abs(topZ - (tz + i)); 
        if (dist < myDist) { myDist = dist; pixel = top; }

        dist = abs(bottomZ - (tz + i)); 
        if (dist < myDist) pixel = bottom; 

        output[id] = pixel; 
    }
 }

__device__ float interpointY(int x1, int y1, int z1, int x2, int y2, int z2, int x0, int z0) 
{
    float xM = (x1 + x2) / 2.0f; 
    float yM = (y1 + y2) / 2.0f; 
    float zM = (z1 + z2) / 2.0f;    
    float nx = x2 - x1; 
    float ny = y2 - y1; 
    float nz = z2 - z1; 

    return yM + (nx * (xM - x0) + nz * (zM - z0)) / ny; 
}

__global__ void kernelMaurerAxis(int *stack, int size, int mod, int bandSize) 
{
    int blkX = blockIdx.x % mod; 
    int band = blockIdx.x / mod; 

    int tx = blkX * blockDim.x + threadIdx.x; 
    int ty = band * bandSize; 
    int tz = blockIdx.y * blockDim.y + threadIdx.y; 

    int lastY = INFINITY; 
    int stackX_1, stackY_1 = INFINITY, stackZ_1, stackX_2, stackY_2 = INFINITY, stackZ_2; 
    int p = MARKER;
    int nx, ny, nz, s1, s2; 
    float i1, i2;     

    for (int i = 0; i < bandSize; i++, ty++) {
        p = tex1Dfetch(pbaTexColor, TOID(tx, ty, tz, size));

        if (p != MARKER) {
            while (stackY_2 != INFINITY) {
                DECODE(s1, stackX_1, stackY_1, stackZ_1); 
                DECODE(s2, stackX_2, stackY_2, stackZ_2); 
                i1 = interpointY(stackX_1, stackY_2, stackZ_1, stackX_2, lastY, stackZ_2, tx, tz); 
                DECODE(p, nx, ny, nz); 
                i2 = interpointY(stackX_2, lastY, stackZ_2, nx, ty, nz, tx, tz); 

                if (i1 < i2) 
                    break;

                lastY = stackY_2; s2 = s1; stackY_2 = stackY_1;

                if (stackY_2 != INFINITY)
                    s1 = stack[TOID(tx, stackY_2, tz, size)]; 
            }

            DECODE(p, nx, ny, nz); 
            s1 = s2; s2 = ENCODE(nx, lastY, nz); 
            stackY_2 = lastY; lastY = ty; 

            stack[TOID(tx, ty, tz, size)] = s2; 
        }
    }

    if (p == MARKER) 
        stack[TOID(tx, ty-1, tz, size)] = ENCODE(INFINITY, lastY, INFINITY); 
}

__global__ void kernelMergeBands(int *stack, short *forward, int size, int mod, int bandSize) 
{
    int blkX = blockIdx.x % mod; 
    int band1 = (blockIdx.x / mod) * 2; 
    int band2 = band1 + 1; 

    int tx = blkX * blockDim.x + threadIdx.x; 
    int tz = blockIdx.y * blockDim.y + threadIdx.y; 

    int firstY, lastY, next, p, id; 
    int3 stack_1, stack_2, current; 
    float i1, i2;

    firstY = band2 * bandSize; 
    lastY = firstY - 1; 
    
    // Band 1, get the two last items
    p = tex1Dfetch(pbaTexLinks, TOID(tx, lastY, tz, size)); 
    DECODE(p, stack_2.x, stack_2.y, stack_2.z); 

    if (stack_2.x == INFINITY) {     // Not a site
        lastY = stack_2.y; 

        if (lastY != INFINITY) {
            p = tex1Dfetch(pbaTexLinks, TOID(tx, lastY, tz, size)); 
            DECODE(p, stack_2.x, stack_2.y, stack_2.z); 
        }
    }   

    if (stack_2.y != INFINITY) {
        p = tex1Dfetch(pbaTexLinks, TOID(tx, stack_2.y, tz, size)); 
        DECODE(p, stack_1.x, stack_1.y, stack_1.z); 
    }

    // Band 2, get the first item
    next = tex1Dfetch(pbaTexPointer, TOID(tx, firstY, tz, size)); 

    if (next < 0)       // Not a site
        firstY = -next; 

    if (firstY != INFINITY) {
        id = TOID(tx, firstY, tz, size); 
        p = tex1Dfetch(pbaTexLinks, id); 
        DECODE(p, current.x, current.y, current.z); 
    }

    int top = 0; 

    while (top < 2 && firstY != INFINITY) {
        while (stack_2.y != INFINITY) {
            i1 = interpointY(stack_1.x, stack_2.y, stack_1.z, stack_2.x, lastY, stack_2.z, tx, tz); 
            i2 = interpointY(stack_2.x, lastY, stack_2.z, current.x, firstY, current.z, tx, tz); 

            if (i1 < i2) 
                break; 

            lastY = stack_2.y; stack_2 = stack_1; 
            top--; 

            if (stack_2.y != INFINITY) {
                p = stack[TOID(tx, stack_2.y, tz, size)]; 
                DECODE(p, stack_1.x, stack_1.y, stack_1.z); 
            }
        }

        // Update pointers to link the current node to the stack
        stack[id] = ENCODE(current.x, lastY, current.z); 

        if (lastY != INFINITY) 
            forward[TOID(tx, lastY, tz, size)] = firstY; 

        top = max(1, top + 1); 

        // Advance the current pointer forward
        stack_1 = stack_2; stack_2 = make_int3(current.x, lastY, current.z); lastY = firstY; 
        firstY = tex1Dfetch(pbaTexPointer, id); 

        if (firstY != INFINITY) {
            id = TOID(tx, firstY, tz, size); 
            p = tex1Dfetch(pbaTexLinks, id); 
            DECODE(p, current.x, current.y, current.z); 
        }
    }

    // Update the head pointer
    firstY = band1 * bandSize; 
    lastY = band2 * bandSize; 

    if (tex1Dfetch(pbaTexPointer, TOID(tx, firstY, tz, size)) == -INFINITY) 
        forward[TOID(tx, firstY, tz, size)] = -fabsf(tex1Dfetch(pbaTexPointer, TOID(tx, lastY, tz, size))); 

    // Update the tail pointer
    firstY = band1 * bandSize + bandSize - 1; 
    lastY = band2 * bandSize + bandSize - 1; 

    p = tex1Dfetch(pbaTexLinks, TOID(tx, lastY, tz, size)); 
    DECODE(p, current.x, current.y, current.z); 

    if (current.x == INFINITY && current.y == INFINITY) {
        p = tex1Dfetch(pbaTexLinks, TOID(tx, firstY, tz, size)); 
        DECODE(p, stack_1.x, stack_1.y, stack_1.z); 

        if (stack_1.x == INFINITY) 
            current.y = stack_1.y; 
        else
            current.y = firstY; 

        stack[TOID(tx, lastY, tz, size)] = ENCODE(current.x, current.y, current.z); 
    }
}

__global__ void kernelCreateForwardPointers(short *output, int size, int mod, int bandSize) 
{
    int blkX = blockIdx.x % mod; 
    int band = blockIdx.x / mod; 

    int tx = blkX * blockDim.x + threadIdx.x; 
    int ty = (band+1) * bandSize - 1; 
    int tz = blockIdx.y * blockDim.y + threadIdx.y; 

    int lasty = INFINITY, nexty; 
    int current, id; 

    // Get the tail pointer
    current = tex1Dfetch(pbaTexLinks, TOID(tx, ty, tz, size)); 

    if (GET_X(current) == INFINITY) 
        nexty = GET_Y(current); 
    else
        nexty = ty; 

    id = TOID(tx, ty, tz, size); 

    for (int i = 0; i < bandSize; i++, ty--, id -= size) 
        if (ty == nexty) {
            output[id] = lasty; 
            nexty = GET_Y(tex1Dfetch(pbaTexLinks, id)); 
            lasty = ty; 
        }
    
    // Store the pointer to the head at the first pixel of this band
    if (lasty != ty + 1) 
        output[id + size] = -lasty; 
}

__global__ void kernelColorAxis(int *output, int size) 
{
    __shared__ int3 s_Stack1[BLOCKX], s_Stack2[BLOCKX];
    __shared__ int s_lastY[BLOCKX]; 
    __shared__ float s_ii[BLOCKX]; 

    int col = threadIdx.x; 
    int tid = threadIdx.y; 
    int tx = blockIdx.x * blockDim.x + col; 
    int tz = blockIdx.y; 

    int3 stack_1, stack_2; 
    int p, lastY; 
    float ii; 

    if (tid == blockDim.y - 1) { 
        lastY = size - 1; 

        p = tex1Dfetch(pbaTexColor, TOID(tx, lastY, tz, size)); 
        DECODE(p, stack_2.x, stack_2.y, stack_2.z); 

        if (stack_2.x == INFINITY) {     // Not a site
            lastY = stack_2.y; 

            if (lastY != INFINITY) {
                p = tex1Dfetch(pbaTexColor, TOID(tx, lastY, tz, size)); 
                DECODE(p, stack_2.x, stack_2.y, stack_2.z); 
            }
        }

        if (stack_2.y != INFINITY) { 
            p = tex1Dfetch(pbaTexColor, TOID(tx, stack_2.y, tz, size)); 
            DECODE(p, stack_1.x, stack_1.y, stack_1.z); 
            ii = interpointY(stack_1.x, stack_2.y, stack_1.z, stack_2.x, lastY, stack_2.z, tx, tz); 
        }

        s_Stack1[col] = stack_1; s_Stack2[col] = stack_2; s_lastY[col] = lastY; s_ii[col] = ii; 
    }

    __syncthreads(); 

    for (int ty = size - 1 - tid; ty >= 0; ty -= blockDim.y) {
        stack_1 = s_Stack1[col]; stack_2 = s_Stack2[col]; lastY = s_lastY[col]; ii = s_ii[col]; 

        while (stack_2.y != INFINITY) {
            if (ty > ii) 
                break; 

            lastY = stack_2.y; stack_2 = stack_1;

            if (stack_2.y != INFINITY) {
                p = tex1Dfetch(pbaTexColor, TOID(tx, stack_2.y, tz, size)); 
                DECODE(p, stack_1.x, stack_1.y, stack_1.z); 

                ii = interpointY(stack_1.x, stack_2.y, stack_1.z, stack_2.x, lastY, stack_2.z, tx, tz); 
            }
        }

        __syncthreads(); 

        output[TOID(tx, ty, tz, size)] = ENCODE(stack_2.x, lastY, stack_2.z); 

        if (tid == blockDim.y - 1) {
            s_Stack1[col] = stack_1; s_Stack2[col] = stack_2; s_lastY[col] = lastY; s_ii[col] = ii; 
        }

        __syncthreads(); 
    }
}

/// A function to draw points and its ID onto GPU texture

__global__ void kerSetPointsInGrid
(
Point3*         pointArr,
int             pointNum,
 int*           inputVoronoi, 
 int            fbosize
)
{
    int tid             = blockIdx.x * blockDim.x + threadIdx.x; 
    int totalThreads    = blockDim.x * gridDim.x; 

    for ( int p = tid; p < pointNum; p += totalThreads ) 
    {
        Point3 pt = pointArr[ p ]; 

        int tx = (int) pt._p[0]; 
        int ty = (int) pt._p[1]; 
        int tz = (int) pt._p[2]; 

        int id = TOID( tx, ty, tz, fbosize ); 

        inputVoronoi[ id ]  = ENCODE( tx, ty, tz ); 
    }

    return;
}

__global__ void kerSetPointIndicesInGrid
(
Point3* pointArr,
int     pointNum,
int*    mapToID,
int     fbosize
)
{
    int tid             = blockIdx.x * blockDim.x + threadIdx.x; 
    int totalThreads    = blockDim.x * gridDim.x; 

    for ( int p = tid; p < pointNum; p += totalThreads ) 
    {
        Point3 pt = pointArr[ p ]; 

        int tx = (int) pt._p[0]; 
        int ty = (int) pt._p[1]; 
        int tz = (int) pt._p[2]; 

        int id          = TOID( tx, ty, tz, fbosize ); 
        mapToID[ id ]   = p; 
    }

    return;
}

// Convert coords to point indices in grid
__global__ void kerSetIndexInGrid
(
    int         gridWidth,
    const int*  pointIndexGrid,
    int*        grid
)
{
    ////
    // Replace coord with index in grid
    ////

    int tid             = blockIdx.x * blockDim.x + threadIdx.x; 
    int totalThreads    = blockDim.x * gridDim.x; 

    // Iterate through voxels

    const int gridIdxMax = gridWidth * gridWidth * gridWidth;

    for ( int gridIdx = tid; gridIdx < gridIdxMax; gridIdx += totalThreads )
    {
        // Coordinate stored in voxel
        const int coordVal = grid[ gridIdx ];
        int3 loc;
        DECODE( coordVal, loc.x, loc.y, loc.z );

        // Convert to point index
        const int locIdx        = TOID( loc.x, loc.y, loc.z, gridWidth );
        const int pointIndex    = pointIndexGrid[ locIdx ];

        // Write index to voxel
        grid[ gridIdx ] = pointIndex;
    }

    return;
}
