gStar4D
=======

gStar4D computes the 3D Delaunay triangulation on the GPU.

The gStar4D algorithm uses neighbourhood information in the 3D digital Voronoi
diagram as an approximation of the 3D Delaunay triangulation. It uses this to
perform massively parallel creation of stars of each input point lifted to 4D
and employs an unique star splaying approach to splay these 4D stars in
parallel and make them consistent.
The result is the 3D Delaunay triangulation of the input constructed fully on
the GPU.

Our CUDA implementation of gStar4D is robust and achieves a speedup of up to 5
times over the 3D Delaunay triangulator of CGAL.

Setup
=====

This project has been primarily tested on Windows 7 64-bit OS using Visual
Studio 2008 and CUDA 4.0.
This is a workstation with Intel i7 2600K CPU and
NVIDIA GTX 580 GPU.

On Windows
----------

Open the GDelaunay solution file using Visual Studio and
compile with the Release build for maximum performance.

On Linux
--------

CMake is needed to build gStar4D on Linux. Make sure that the CUDA
capability values in the CMakeLists.txt file matches that of your GPU.

To build and execute using CMake:

    $ cd gstar4d
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    $ ./gstar4d --help

Documentation
=============

Chapter 6 of my [PhD thesis](gdel3d_thesis.pdf) describes the details of the
gStar4D algorithm.

Reference
=========

To refer to gStar4D, please use:

    Ashwin Nanjappa, "Delaunay triangulation in R3 on the GPU", PhD thesis,
    School of Computing, National University of Singapore, 2012.

To cite gStar4D, please use this BibTeX entry:

    @phdthesis{Ashwin2012GPUDelaunay,
        author = {Nanjappa, Ashwin},
        school = {National University of Singapore},
        title = {Delaunay triangulation in R3 on the {GPU}},
        year = {2012}
    }

License
=======

Author: Ashwin Nanjappa

Project: gStar4D

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
