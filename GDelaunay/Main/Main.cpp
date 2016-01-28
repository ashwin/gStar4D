/*
Author: Ashwin Nanjappa
Filename: Main.cpp

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
//                                     Main
////////////////////////////////////////////////////////////////////////////////

#include "Application.h"
#include "Config.h"

int main( int argc, const char* argv[] )
{
    ////
    // Below are the default configuration values.
    // Note: Do NOT change them here for experimentation!
    //       Instead use their command-line parameters.
    ////

    Config& config          = getConfig();
    config._seed            = 0;
    config._runNum          = 1;
    config._gridSize        = 256;
    config._pointNum        = 100000;
    config._dist            = UniformDistribution;
    config._facetMax        = -1;   // Set below
    config._logVerbose      = false;
    config._logStats        = false;
    config._logTiming       = false;
    config._doCheck         = false;
    config._inFile          = false;
    config._inFilename      = "";
    config._predThreadNum   = -1;
    config._predBlockNum    = -1;
    config._outFile         = false;
    config._outFilename     = "";
    config._doSorting       = true;

    ////
    // Parse input arguments
    ////

    int idx = 1;
    bool printHelp = false;

    while ( idx < argc )
    {
        if ( 0 == string( "-n" ).compare( argv[ idx ] ) )
        {
            config._pointNum = atoi( argv[ idx + 1 ] ); 
            ++idx; 
        }
        else if ( 0 == string( "-g" ).compare( argv[ idx ] ) )
        {
            config._gridSize = atoi( argv[ idx + 1 ] ); 
            ++idx; 
        }
        else if ( 0 == string( "-r" ).compare( argv[ idx ] ) )
        {
            config._runNum = atoi( argv[ idx + 1 ] ); 
            ++idx; 
        }
        else if ( 0 == string( "-s" ).compare( argv[ idx ] ) )
        {
            config._seed = atoi( argv[ idx + 1 ] ); 
            ++idx; 
        }
        else if ( 0 == string( "-f" ).compare( argv[ idx ] ) )
        {
            config._facetMax = atoi( argv[ idx + 1 ] ); 
            ++idx; 
        }
        else if ( 0 == string( "-d" ).compare( argv[ idx ] ) )
        {
            const int distVal   = atoi( argv[ idx + 1 ] );
            config._dist        = ( Distribution ) distVal;

            ++idx; 
        }
        else if ( 0 == string( "-verbose" ).compare( argv[ idx ] ) )
        {
            config._logVerbose = true;
        }
        else if ( 0 == string( "-stats" ).compare( argv[ idx ] ) )
        {
            config._logStats = true;
        }
        else if ( 0 == string( "-timing" ).compare( argv[ idx ] ) )
        {
            config._logTiming = true;
        }
        else if ( 0 == string( "-check" ).compare( argv[ idx ] ) )
        {
            config._doCheck = true;
        }
        else if ( 0 == string( "-inFile" ).compare( argv[ idx ] ) )
        {
            config._inFile      = true;
            config._inFilename  = string( argv[ idx + 1 ] );

            ++idx;
        }
        else if ( 0 == string( "-predThreadNum" ).compare( argv[ idx ] ) )
        {
            config._predThreadNum = atoi( argv[ idx + 1 ] ); 
            ++idx; 
        }
        else if ( 0 == string( "-predBlockNum" ).compare( argv[ idx ] ) )
        {
            config._predBlockNum = atoi( argv[ idx + 1 ] ); 
            ++idx; 
        }
        else if ( 0 == string( "-outFile" ).compare( argv[ idx ] ) )
        {
            config._outFile     = true;
            config._outFilename = string( argv[ idx + 1 ] );

            ++idx;
        }
        else if ( 0 == string( "-sort" ).compare( argv[ idx ] ) )
        {
            config._doSorting = true;
        }
        else if ( 0 == string( "--help" ).compare( argv[ idx ] ) )
        {
            printHelp = true;
        }
        else if ( 0 == string( "-h" ).compare( argv[ idx ] ) )
        {
            printHelp = true;
        }
        else
        {
            cout << "Error in input argument: " << argv[ idx ] << endl << endl;
            printHelp = true;
        }

        if (printHelp)
        {

            cout << "Syntax: gstar4d [-n PointNum] [-g GridSize] [-s Seed] [-r Max] [-d Distribution] [-f FacetMax] [-verbose] [-stats] [-timing] [-check]" << endl; 
            cout << "Distribution: 0 - Uniform, 1 - Gaussian, 2 - Ball, 3 - Sphere, 4 - Grid" << endl;
            exit(1);
        }

        ++idx; 
    }

    // Adjust facetMax
    if ( -1 == config._facetMax )
    {
        // Note: Change these for different VRAM size
        config._facetMax = ( UniformDistribution == config._dist ) ? 12000000 : 10000000;
    }

    ////
    // Print current configuration
    ////

    cout << "GDelaunay ...";
    cout << " Seed: "   << config._seed;
    cout << " RunNum: " << config._runNum;
    cout << " Dist: "   << config._dist;
    cout << " Grid: "   << config._gridSize;

    if ( config._inFile ) cout << " Input: " << config._inFilename;
    else                  cout << " Points: " << config._pointNum;

    cout << endl;

    ////
    // Compute Delaunay
    ////

    Application app;
    app.doRun();

    return 0;
}

Config& getConfig()
{
    static Config _config;
    return _config;
}

////////////////////////////////////////////////////////////////////////////////
