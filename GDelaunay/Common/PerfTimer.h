/*
Author: Ashwin Nanjappa
Filename: PerfTimer.h

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

///////////////////////////////////////////////////////////////////////////////
// PerfTimer: A high performance timer class
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "CudaWrapper.h"
#include "STLWrapper.h"

#ifdef _WIN32

#define NOMINMAX
#include <windows.h>

struct PerfTimer
{
    float         _freq;
    LARGE_INTEGER _startTime;
    LARGE_INTEGER _stopTime;

    PerfTimer()
    {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        _freq = 1.0f / freq.QuadPart;
    }

    void start()
    {
        QueryPerformanceCounter(&_startTime);
    }

    void stop()
    {
        QueryPerformanceCounter(&_stopTime);
    }

    double value() const
    {
        return (_stopTime.QuadPart - _startTime.QuadPart) * _freq;
    }
};

#else

#include <sys/time.h>

const long long NANO_PER_SEC = 1000000000LL;
const long long MICRO_TO_NANO = 1000LL;

struct PerfTimer
{
    long long _startTime;
    long long _stopTime;

    long long _getTime()
    {
        struct timeval tv;
        long long ntime;

        const int ret = gettimeofday(&tv, NULL);
        assert(0 == ret);

        ntime  = NANO_PER_SEC;
        ntime *= tv.tv_sec;
        ntime += tv.tv_usec * MICRO_TO_NANO;

        return ntime;
    }

    void start()
    {
        _startTime = _getTime();
    }

    void stop()
    {
        _stopTime = _getTime();
    }

    double value() const
    {
        return ((double) _stopTime - _startTime) / NANO_PER_SEC;
    }
};
#endif

class HostTimer : public PerfTimer
{
public:
    void start()
    {
        CudaSafeCall(cudaDeviceSynchronize());
        PerfTimer::start();
    }

    void stop()
    {
        CudaSafeCall(cudaDeviceSynchronize());
        PerfTimer::stop();
    }

    double value()
    {
        return PerfTimer::value() * 1000;
    }

    void print(const string outStr = "")
    {
        cout << "Time: " << value() << " for " << outStr << endl;
    }
};

///////////////////////////////////////////////////////////////////////////////
