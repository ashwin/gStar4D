/*
Author: Ashwin Nanjappa
Filename: Application.cpp

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
//                                 Application
////////////////////////////////////////////////////////////////////////////////

// Self
#include "Application.h"

// Project
#include "Config.h"
#include "DtRandom.h"
#include "GDelaunay.h"
#include "STLWrapper.h"
#include "PerfTimer.h"

// Obtained from: C:\ProgramData\NVIDIA Corporation\GPU SDK\C\common\inc\cutil_inline_runtime.h
// This function returns the best GPU (with maximum GFLOPS)
int cutGetMaxGflopsDeviceId()
{
    int current_device   = 0, sm_per_multiproc = 0;
    int max_compute_perf = 0, max_perf_device  = 0;
    int device_count     = 0, best_SM_arch     = 0;
    int arch_cores_sm[3] = { 1, 8, 32 };
    cudaDeviceProp deviceProp;

    cudaGetDeviceCount( &device_count );
    // Find the best major SM Architecture GPU device
    while ( current_device < device_count ) {
        cudaGetDeviceProperties( &deviceProp, current_device );
        if (deviceProp.major > 0 && deviceProp.major < 9999)
        {
            if ( deviceProp.major > best_SM_arch )
                best_SM_arch = deviceProp.major;
        }
        current_device++;
    }

    // Find the best CUDA capable GPU device
    current_device = 0;
    while( current_device < device_count ) {
        cudaGetDeviceProperties( &deviceProp, current_device );
        if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
            sm_per_multiproc = 1;
        } else if (deviceProp.major <= 2) {
            sm_per_multiproc = arch_cores_sm[deviceProp.major];
        } else {
            sm_per_multiproc = arch_cores_sm[2];
        }

        int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
        if( compute_perf  > max_compute_perf ) {
            // If we find GPU with SM major > 2, search only these
            if ( best_SM_arch > 2 ) {
                // If our device==dest_SM_arch, choose this, or else pass
                if (deviceProp.major == best_SM_arch) { 
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
                }
            } else {
                max_compute_perf  = compute_perf;
                max_perf_device   = current_device;
            }
        }
        ++current_device;
    }
    return max_perf_device;
}

void Application::doRun()
{
    Config& config = getConfig();

    ////
    // Run Delaunay iterations
    ////

    double initTimeSum  = 0.0;
    double pbaTimeSum   = 0.0;
    double starTimeSum  = 0.0;
    double splayTimeSum = 0.0;
    double outTimeSum   = 0.0;
    double totTimeSum   = 0.0;
    DoubleDeq   timeDeq;

    for ( int i = 0; i < config._runNum; ++i )
    {
        double initTime  = 0.0;
        double pbaTime   = 0.0;
        double starTime  = 0.0;
        double splayTime = 0.0;
        double outTime   = 0.0;
        double totTime   = 0.0;

        _init();
        _doRun( initTime, pbaTime, starTime, splayTime, outTime, totTime );
        _deInit();

        initTimeSum  += initTime;
        pbaTimeSum   += pbaTime;
        starTimeSum  += starTime;
        splayTimeSum += splayTime;
        outTimeSum   += outTime;
        totTimeSum   += totTime;

        timeDeq.push_back( totTime );

        ++config._seed;
    }

    ////
    // Compute mean and deviation
    ////

    const int timeNum   = ( int ) timeDeq.size();
    double timeSum      = 0;

    for ( int i = 0; i < timeNum; ++i )
        timeSum += timeDeq[i];

    const double timeAvg = timeSum / timeNum;

    double timeSdev = 0;

    for ( int i = 0; i < timeNum; ++i )
        timeSdev += ( timeDeq[i] - timeAvg ) * ( timeDeq[i] - timeAvg );

    timeSdev = sqrt( timeSdev / timeNum );

    // Write time to file

    ofstream logFile( "gStar4D-time.txt", ofstream::app );
    assert( logFile && "Could not open time file!" );

    logFile << "GridSize,"   << config._gridSize << ",";
    logFile << "Runs,"       << config._runNum   << ",";
    logFile << "Points,"     << config._pointNum << ",";
    logFile << "Input,"      << ( config._inFile ? config._inFilename : DistStr[ config._dist ] ) << ",";
    logFile << "Total Time," << totTimeSum   / ( config._runNum * 1000.0 ) << ",";
    logFile << "Init Time,"  << initTimeSum  / ( config._runNum * 1000.0 ) << ",";
    logFile << "PBA Time,"   << pbaTimeSum   / ( config._runNum * 1000.0 ) << ",";
    logFile << "Star Time,"  << starTimeSum  / ( config._runNum * 1000.0 ) << ",";
    logFile << "Splay Time," << splayTimeSum / ( config._runNum * 1000.0 ) << ",";
    logFile << "Out Time,"   << outTimeSum   / ( config._runNum * 1000.0 ) << endl;

    // Write counts to file

    if ( config._logStats )
    {
        extern int LogStarInsNum;
        extern int LogSplayInsNum;
        extern int LogLoopNum;

        ofstream oFile( "gStar4D-count.txt", ofstream::app );
        assert( oFile && "Could not open count file!" );

        oFile << "GridSize,"   << config._gridSize << ",";
        oFile << "Runs,"       << config._runNum   << ",";
        oFile << "Points,"     << config._pointNum << ",";
        oFile << "Input,"      << ( config._inFile ? config._inFilename : DistStr[ config._dist ] ) << ",";

        oFile << "Total insertions," << LogStarInsNum + LogSplayInsNum << ",";
        oFile << "Star insertions,"  << LogStarInsNum  << ",";
        oFile << "Splay insertions," << LogSplayInsNum << ",";
        oFile << "Loop number,"      << LogLoopNum     << endl;
    }

    return;
}

void Application::_init()
{
    // Pick the best CUDA device
    const int deviceIdx = cutGetMaxGflopsDeviceId();
    CudaSafeCall( cudaSetDevice( deviceIdx ) );

    // Default kernel configuration
    CudaSafeCall( cudaDeviceSetCacheConfig( cudaFuncCachePreferShared ) );

    return;
}

void Application::_doRun
(
double& initTime,
double& pbaTime,
double& starTime,
double& splayTime,
double& outTime,
double& totTime
)
{
    Config& config = getConfig();

    cout << "Seed: " << config._seed << endl;

    // Generate points
    if ( config._inFile ) readPoints();
    else                  makePoints();

    if ( config._outFile )
    {
        const string outFilename    = config._inFile ? "Input-Scaled-Points.asc" : "Input-Points.asc";
        const Point3HVec& pointVec  = config._inFile ? _scaledVec : _pointVec;

        ofstream outFile( outFilename.c_str() );

        for ( int pi = 0; pi < ( int ) pointVec.size(); ++pi )
        {
            const Point3& pt = pointVec[ pi ];
            outFile << pt._p[0] << " " << pt._p[1] << " " << pt._p[2] << endl;
        }
    }

    ////
    // Compute Delaunay
    ////

    assert( ( config._inFile == !_scaledVec.empty() ) && "Scaled points should exist for points read from file!" );

    HostTimer initTimer;
    HostTimer timerAll;

    timerAll.start();
    {
        initTimer.start();
        {
            gdelInit( config, _pointVec, _scaledVec );
        }
        initTimer.stop();

        gdelCompute( pbaTime, starTime, splayTime, outTime );
    }
    timerAll.stop();

    initTime = initTimer.value();
    totTime  = timerAll.value();

    cout << endl;
    cout << "Init:        " << initTime  << endl;
    cout << "PBA:         " << pbaTime   << endl; 
    cout << "InitStar:    " << starTime  << endl;
    cout << "Consistency: " << splayTime << endl;
    cout << "StarOutput:  " << outTime   << endl << endl;
    cout << "Total Time:  " << totTime   << endl;

    ////
    // Check
    ////

    if ( config._doCheck || config._outFile )
    {
        TetraMesh tetMesh;
        tetMesh.setPoints( _pointVec );
        getTetraFromGpu( tetMesh );

        if ( config._doCheck )
        {
            tetMesh.check();
        }

        if ( config._outFile )
        {
            tetMesh.writeToFile( config._outFilename );
        }
    }

    ////
    // Destroy
    ////

    gdelDeInit();
    _pointVec.clear();
    _scaledVec.clear();

    return;
}

void Application::_deInit()
{
    CudaSafeCall( cudaDeviceReset() );

    return;
}

void Application::makePoints()
{
    Config& config = getConfig();

    assert( _pointVec.empty() && "Input point vector not empty!" );

    ////
    // Initialize seed
    ////

    // Points in range [1..width-2]
    const int minWidth = 1;
    const int maxWidth = config._gridSize - 2;

    DtRandom randGen;

    switch ( config._dist )
    {
    case UniformDistribution:
    case GaussianDistribution:
    case GridDistribution:
        randGen.init( config._seed, minWidth, maxWidth );
        break;
    case BallDistribution:
    case SphereDistribution:
        randGen.init( config._seed, 0, 1 );
        break;
    default:
        assert( false );
        break;
    }

    ////
    // Generate points
    ////
    
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Point3Set pointSet;

    for ( int i = 0; i < config._pointNum; ++i )
    {
        bool uniquePoint = false;

        // Loop until we get unique point
        while ( !uniquePoint )
        {
            switch ( config._dist )
            {
            case UniformDistribution:
                {
                    x = randGen.getNext();
                    y = randGen.getNext();
                    z = randGen.getNext();
                }
                break;

            case GaussianDistribution:
                {
                    randGen.nextGaussian( x, y, z);
                }
                break;

            case BallDistribution:
                {
                    float d;

                    do
                    {
                        x = randGen.getNext() - 0.5f; 
                        y = randGen.getNext() - 0.5f; 
                        z = randGen.getNext() - 0.5f; 
                    
                        d = x * x + y * y + z * z;
                        
                    } while ( d > 0.45f * 0.45f );

                    x += 0.5f;
                    y += 0.5f;
                    z += 0.5f;
                    x *= maxWidth;
                    y *= maxWidth;
                    z *= maxWidth;
                }
                break;

            case SphereDistribution:
                {
                    float d;

                    do
                    {
                        x = randGen.getNext() - 0.5f; 
                        y = randGen.getNext() - 0.5f; 
                        z = randGen.getNext() - 0.5f; 
                    
                        d = x * x + y * y + z * z;
                        
                    } while ( d > ( 0.45f * 0.45f ) || d < ( 0.4f * 0.4f ) );

                    x += 0.5f;
                    y += 0.5f;
                    z += 0.5f;
                    x *= maxWidth;
                    y *= maxWidth;
                    z *= maxWidth;
                }
                break;

            case GridDistribution:
                {
                    float v[3];

                    for ( int i = 0; i < 3; ++i )
                    {
                        const float val     = randGen.getNext();
                        const float frac    = val - floor( val );
                        v[ i ]              = ( frac < 0.5f ) ? floor( val ) : ceil( val );
                    }

                    x = v[0];
                    y = v[1];
                    z = v[2];
                }
                break;
                
            default:
                {
                    assert( false );
                }
                break;
            }

            // Adjust to bounds
            if ( floor( x ) >= maxWidth )   x -= 1.0f;
            if ( floor( y ) >= maxWidth )   y -= 1.0f;
            if ( floor( z ) >= maxWidth )   z -= 1.0f;

            assert( ( x >= minWidth ) && ( x <= maxWidth ) );
            assert( ( y >= minWidth ) && ( y <= maxWidth ) );
            assert( ( z >= minWidth ) && ( z <= maxWidth ) );

            const Point3 point = { x, y, z };

            if ( pointSet.end() == pointSet.find( point ) )
            {
                pointSet.insert( point );
                _pointVec.push_back( point );

                uniquePoint = true;
            }
        }
    }

    return;
}

void Application::readPoints()
{
    assert( _pointVec.empty() && "Input point vector not empty!" );

    Config& config = getConfig();
    ifstream inFile( config._inFilename.c_str() );

    if ( !inFile )
    {
        cout << "Error opening input file: " << config._inFilename << " !!!" << endl;
        exit( 1 );
    }
    else
    {
        cout << "Reading from point file ..." << endl;
    }

    ////
    // Read input points
    ////

    string strVal;
    Point3 point;
    Point3Set pointSet;
    int idx         = 0;
    int orgCount    = 0;
    float val       = 0.0f;
    float minVal    = 999.0f;
    float maxVal    = -999.0f;

    while ( inFile >> strVal )
    {
        istringstream iss( strVal );

        // Read a coordinate
        iss >> val;
        point._p[ idx ] = val;
        ++idx;

        // Compare bounds
        if ( val < minVal ) minVal = val;
        if ( val > maxVal ) maxVal = val;

        // Read a point
        if ( 3 == idx )
        {
            idx = 0;

            ++orgCount;

            // Check if point unique
            if ( pointSet.end() == pointSet.find( point ) )
            {
                pointSet.insert( point );
                _pointVec.push_back( point );
            }
        }
    }

    ////
    // Check for duplicate points
    ////

    const int dupCount = orgCount - _pointVec.size();

    if ( dupCount > 0 )
    {
        cout << dupCount << " duplicate points in input file!" << endl;
    }

    if ( config._logStats )
    {
        cout << "Min: " << minVal << " Max: " << maxVal;
        cout << " Min-Scaled: " << _scalePoint( ( float ) config._gridSize, minVal, maxVal, minVal );
        cout << " Max-Scaled: " << _scalePoint( ( float ) config._gridSize, minVal, maxVal, maxVal ) << endl;
    }

    ////
    // Scale points
    ////

    pointSet.clear();

    // Iterate input points
    for ( int ip = 0; ip < ( int ) _pointVec.size(); ++ip )
    {
        Point3& inPt = _pointVec[ ip ];

        // Iterate coordinates
        for ( int vi = 0; vi < 3; ++vi )
        {
            const RealType inVal    = inPt._p[ vi ];
            const RealType outVal   = _scalePoint( ( RealType ) config._gridSize, minVal, maxVal, inVal );
            inPt._p[ vi ]           = outVal;
        }

        // Check if point unique
        if ( pointSet.end() == pointSet.find( inPt ) )
        {
            pointSet.insert( inPt );
            _scaledVec.push_back( inPt );
        }
    }

    assert( ( _pointVec.size() == _scaledVec.size() ) && "Duplicate points created due to scaling!!!" );

    config._pointNum = _scaledVec.size();

    if ( config._logVerbose )
        cout << "Points read from file: " << config._pointNum << endl;

    return;
}

RealType Application::_scalePoint( RealType gridWidth, float minVal, float maxVal, RealType inVal )
{
    // Translate
    inVal = inVal - minVal; // MinVal can be -ve
    assert( inVal >= 0 );

    // Scale
    const float rangeVal = maxVal - minVal;
    inVal = ( gridWidth - 3.0f ) * inVal / rangeVal;
    assert( inVal <= ( gridWidth - 2 ) );

    inVal += 1.0f;

    return inVal;
}

////////////////////////////////////////////////////////////////////////////////
