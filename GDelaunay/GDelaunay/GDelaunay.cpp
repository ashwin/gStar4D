/*
Author: Ashwin Nanjappa
Filename: GDelaunay.cpp

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
//                               GPU Main Code
////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////// Headers //

// Project
#include "Config.h"
#include "GDelHost.h"
#include "PerfTimer.h"
#include "Pba.h"

////////////////////////////////////////////////////////////////////////////////

void gdelInit
(
const Config&   config,
Point3HVec&     pointHVec,
Point3HVec&     scaledHVec
)
{
    Point3DVec* pointDVec;

    starsInit( pointHVec, scaledHVec, config, &pointDVec );
    pbaInit( config._gridSize, pointDVec );

    return;
}

void gdelCompute( double& timePba, double& timeInitialStar, double& timeConsistent, double& timeOutput )
{
    HostTimer timer0;

    timer0.start();
        int* grid;
        doPba( &grid );
    timer0.stop();
    timePba = timer0.value();

    timer0.start();
        makeStarsFromGrid( grid );
    timer0.stop();
    timeInitialStar = timer0.value();

    timer0.start();
        processFacets();
    timer0.stop();
    timeConsistent = timer0.value();

    timer0.start();
        makeTetraFromStars();
    timer0.stop();
    timeOutput = timer0.value();

    return;
}

void getTetraFromGpu( TetraMesh& tetraMesh )
{
    const TetraHVec& hostTetVec = getHostTetra();
    tetraMesh.setTetra( hostTetVec );

    return;
}

void gdelDeInit()
{
    pbaDeinit();
    starsDeinit();

    return;
}

////////////////////////////////////////////////////////////////////////////////
