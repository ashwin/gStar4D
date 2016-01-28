/*
Author: Ashwin Nanjappa
Filename: CudaWrapper.h

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
//                                 CUDA Wrapper
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////

// Define this to enable error checking
#define CUDA_CHECK_ERROR

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_CHECK_ERROR

#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);

    do
    {
        if ( cudaSuccess != err )
        {
            fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                     file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }
    } while ( 0 );

#pragma warning( pop )

#endif  // CUDA_CHECK_ERROR

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_CHECK_ERROR

#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);

    do
    {
        cudaError err = cudaGetLastError();
        if ( cudaSuccess != err )
        {
            fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                     file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }

        // More careful checking. However, this will affect performance.
        // Comment away if needed.
        err = cudaDeviceSynchronize();
        if( cudaSuccess != err )
        {
            fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                     file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }
    } while ( 0 );

#pragma warning( pop )

#endif // CUDA_CHECK_ERROR

    return;
}

#if __CUDA_ARCH__ >= 200 && defined( CUDA_CHECK_ERROR )
#define CudaAssert(X)                                           \
    if ( !(X) )                                                 \
    {                                                           \
        printf( "!!!Thread %d:%d failed assert at %s:%d!!!\n",  \
            blockIdx.x, threadIdx.x, __FILE__, __LINE__ );      \
    } 
#else
#define CudaAssert(X) 
#endif

template< typename T >
T* cuNew( int num )
{
    T* loc              = NULL;
    const size_t space  = num * sizeof( T );
    CudaSafeCall( cudaMalloc( &loc, space ) );

    return loc;
}

template< typename T >
void cuDelete( T** loc )
{
    CudaSafeCall( cudaFree( *loc ) );
    *loc = NULL;
    return;
}

inline void cuPrintMemory( const char* inStr )
{
    const int MegaByte = ( 1 << 20 );

    size_t free;
    size_t total;
    CudaSafeCall( cudaMemGetInfo( &free, &total ) );

    printf( "[%s] Memory used: %d MB\n", inStr, (int) ( total - free ) / MegaByte );

    return;
}

////////////////////////////////////////////////////////////////////////////////
