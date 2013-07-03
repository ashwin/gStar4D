/*
Author: Ashwin Nanjappa and Cao Thanh Tung
Filename: DtRandom.cpp

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
//                           Random Number Generator
////////////////////////////////////////////////////////////////////////////////

// Self
#include "DtRandom.h"

// External
#include <climits>
#include <cmath>

void DtRandom::init( int seed, int minVal, int maxVal )
{
    _min    = ( float ) minVal;
    _max    = ( float ) maxVal;

    // Seeds
    _z      = seed;
    _w      = seed;
    _jsr    = seed;
    _jcong  = seed;

    return;
}

unsigned long DtRandom::znew() 
{ return (_z = 36969 * (_z & 0xfffful) + (_z >> 16)); };

unsigned long DtRandom::wnew() 
{ return (_w = 18000 * (_w & 0xfffful) + (_w >> 16)); };

unsigned long DtRandom::MWC()  
{ return ((znew() << 16) + wnew()); };

unsigned long DtRandom::SHR3()
{ _jsr ^= (_jsr << 17); _jsr ^= (_jsr >> 13); return (_jsr ^= (_jsr << 5)); };

unsigned long DtRandom::CONG() 
{ return (_jcong = 69069 * _jcong + 1234567); };

unsigned long DtRandom::rand_int()         // [0,2^32-1]
{ return ((MWC() ^ CONG()) + SHR3()); };

float DtRandom::random()     // [0,1)
{ return ((float) rand_int() / (float(ULONG_MAX)+1)); };

float DtRandom::getNext()
{
    const float val = _min + ( _max - _min) * random(); 
    return val; 
}

void DtRandom::nextGaussian(float &x, float &y, float &z)
{
    float x1, x2, x3, w;
    float tx, ty, tz; 

    do {
        do {
            x1 = 2.0f * random() - 1.0f;
            x2 = 2.0f * random() - 1.0f;
            x3 = 2.0f * random() - 1.0f;
            w = x1 * x1 + x2 * x2 + x3 * x3;
        } while ( w >= 1.0 );

        w = sqrt( (-2.0f * log( w ) ) / w );
        tx = x1 * w;
        ty = x2 * w;
        tz = x3 * w; 
    } while (tx < -3 || tx >= 3 || ty < -3 || ty >= 3 || tz < -3 || tz >= 3);

    x = _min + (_max - _min) * ( (tx + 3.0f) / 6.0f );
    y = _min + (_max - _min) * ( (ty + 3.0f) / 6.0f );
    z = _min + (_max - _min) * ( (tz + 3.0f) / 6.0f ); 

    return;
}

////////////////////////////////////////////////////////////////////////////////
