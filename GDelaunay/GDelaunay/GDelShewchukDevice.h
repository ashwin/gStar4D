/*
Author: Ashwin Nanjappa and Cao Thanh Tung
Filename: GDelShewchukDevice.h

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
//                       Shewchuk Predicates on CUDA
////////////////////////////////////////////////////////////////////////////////

#pragma once

/*****************************************************************************/
/*                                                                           */
/*  Routines for Arbitrary Precision Floating-point Arithmetic               */
/*  and Fast Robust Geometric Predicates                                     */
/*  (predicates.c)                                                           */
/*                                                                           */
/*  May 18, 1996                                                             */
/*                                                                           */
/*  Placed in the public domain by                                           */
/*  Jonathan Richard Shewchuk                                                */
/*  School of Computer Science                                               */
/*  Carnegie Mellon University                                               */
/*  5000 Forbes Avenue                                                       */
/*  Pittsburgh, Pennsylvania  15213-3891                                     */
/*  jrs@cs.cmu.edu                                                           */
/*                                                                           */
/*  This file contains C implementation of algorithms for exact addition     */
/*    and multiplication of floating-point numbers, and predicates for       */
/*    robustly performing the orientation and incircle tests used in         */
/*    computational geometry.  The algorithms and underlying theory are      */
/*    described in Jonathan Richard Shewchuk.  "Adaptive Precision Floating- */
/*    Point Arithmetic and Fast Robust Geometric Predicates."  Technical     */
/*    Report CMU-CS-96-140, School of Computer Science, Carnegie Mellon      */
/*    University, Pittsburgh, Pennsylvania, May 1996.  (Submitted to         */
/*    Discrete & Computational Geometry.)                                    */
/*                                                                           */
/*  This file, the paper listed above, and other information are available   */
/*    from the Web page http://www.cs.cmu.edu/~quake/robust.html .           */
/*                                                                           */
/*****************************************************************************/

/*****************************************************************************/
/*                                                                           */
/*  Using this code:                                                         */
/*                                                                           */
/*  First, read the short or long version of the paper (from the Web page    */
/*    above).                                                                */
/*                                                                           */
/*  Be sure to call exactinit() once, before calling any of the arithmetic   */
/*    functions or geometric predicates.  Also be sure to turn on the        */
/*    optimizer when compiling this file.                                    */
/*                                                                           */
/*                                                                           */
/*  Several geometric predicates are defined.  Their parameters are all      */
/*    points.  Each point is an array of two or three floating-point         */
/*    numbers.  The geometric predicates, described in the papers, are       */
/*                                                                           */
/*    orient2d(pa, pb, pc)                                                   */
/*    orient2dfast(pa, pb, pc)                                               */
/*    orient3d(pa, pb, pc, pd)                                               */
/*    orient3dfast(pa, pb, pc, pd)                                           */
/*    incircle(pa, pb, pc, pd)                                               */
/*    incirclefast(pa, pb, pc, pd)                                           */
/*    insphere(pa, pb, pc, pd, pe)                                           */
/*    inspherefast(pa, pb, pc, pd, pe)                                       */
/*                                                                           */
/*  Those with suffix "fast" are approximate, non-robust versions.  Those    */
/*    without the suffix are adaptive precision, robust versions.  There     */
/*    are also versions with the suffices "exact" and "slow", which are      */
/*    non-adaptive, exact arithmetic versions, which I use only for timings  */
/*    in my arithmetic papers.                                               */
/*                                                                           */
/*                                                                           */
/*  An expansion is represented by an array of floating-point numbers,       */
/*    sorted from smallest to largest magnitude (possibly with interspersed  */
/*    zeros).  The length of each expansion is stored as a separate integer, */
/*    and each arithmetic function returns an integer which is the length    */
/*    of the expansion it created.                                           */
/*                                                                           */
/*  Several arithmetic functions are defined.  Their parameters are          */
/*                                                                           */
/*    e, f           Input expansions                                        */
/*    elen, flen     Lengths of input expansions (must be >= 1)              */
/*    h              Output expansion                                        */
/*    b              Input scalar                                            */
/*                                                                           */
/*  The arithmetic functions are                                             */
/*                                                                           */
/*    grow_expansion(elen, e, b, h)                                          */
/*    grow_expansion_zeroelim(elen, e, b, h)                                 */
/*    expansion_sum(elen, e, flen, f, h)                                     */
/*    expansion_sum_zeroelim1(elen, e, flen, f, h)                           */
/*    expansion_sum_zeroelim2(elen, e, flen, f, h)                           */
/*    fast_expansion_sum(elen, e, flen, f, h)                                */
/*    fast_expansion_sum_zeroelim(elen, e, flen, f, h)                       */
/*    linear_expansion_sum(elen, e, flen, f, h)                              */
/*    linear_expansion_sum_zeroelim(elen, e, flen, f, h)                     */
/*    scale_expansion(elen, e, b, h)                                         */
/*    scale_expansion_zeroelim(elen, e, b, h)                                */
/*    compress(elen, e, h)                                                   */
/*                                                                           */
/*  All of these are described in the long version of the paper; some are    */
/*    described in the short version.  All return an integer that is the     */
/*    length of h.  Those with suffix _zeroelim perform zero elimination,    */
/*    and are recommended over their counterparts.  The procedure            */
/*    fast_expansion_sum_zeroelim() (or linear_expansion_sum_zeroelim() on   */
/*    processors that do not use the round-to-even tiebreaking rule) is      */
/*    recommended over expansion_sum_zeroelim().  Each procedure has a       */
/*    little note next to it (in the code below) that tells you whether or   */
/*    not the output expansion may be the same array as one of the input     */
/*    expansions.                                                            */
/*                                                                           */
/*                                                                           */
/*  If you look around below, you'll also find macros for a bunch of         */
/*    simple unrolled arithmetic operations, and procedures for printing     */
/*    expansions (commented out because they don't work with all C           */
/*    compilers) and for generating random floating-point numbers whose      */
/*    significand bits are all random.  Most of the macros have undocumented */
/*    requirements that certain of their parameters should not be the same   */
/*    variable; for safety, better to make sure all the parameters are       */
/*    distinct variables.  Feel free to send email to jrs@cs.cmu.edu if you  */
/*    have questions.                                                        */
/*                                                                           */
/*****************************************************************************/

/* On some machines, the exact arithmetic routines might be defeated by the  */
/*   use of internal extended precision floating-point registers.  Sometimes */
/*   this problem can be fixed by defining certain values to be volatile,    */
/*   thus forcing them to be stored to memory and rounded off.  This isn't   */
/*   a great solution, though, as it slows the arithmetic down.              */
/*                                                                           */
/* To try this out, write "#define INEXACT volatile" below.  Normally,       */
/*   however, INEXACT should be defined to be nothing.  ("#define INEXACT".) */

#define INEXACT                          /* Nothing */
//#define INEXACT volatile

#ifdef REAL_TYPE_DOUBLE
#define REAL double                      /* float or double */
#define Absolute(a)  fabs(a) 
#define MUL(a,b)    __dmul_rn(a,b)
#else
#define REAL float
#define Absolute(a)  fabsf(a) 
#define MUL(a,b)    __fmul_rn(a,b)
#endif

/* Which of the following two methods of finding the absolute values is      */
/*   fastest is compiler-dependent.  A few compilers can inline and optimize */
/*   the fabs() call; but most will incur the overhead of a function call,   */
/*   which is disastrously slow.  A faster way on IEEE machines might be to  */
/*   mask the appropriate bit, but that's difficult to do in C.              */

//#define Absolute(a)  ((a) >= 0.0 ? (a) : -(a))


/* Many of the operations are broken up into two pieces, a main part that    */
/*   performs an approximate operation, and a "tail" that computes the       */
/*   roundoff error of that operation.                                       */
/*                                                                           */
/* The operations Fast_Two_Sum(), Fast_Two_Diff(), Two_Sum(), Two_Diff(),    */
/*   Split(), and Two_Product() are all implemented as described in the      */
/*   reference.  Each of these macros requires certain variables to be       */
/*   defined in the calling routine.  The variables `bvirt', `c', `abig',    */
/*   `_i', `_j', `_k', `_l', `_m', and `_n' are declared `INEXACT' because   */
/*   they store the result of an operation that may incur roundoff error.    */
/*   The input parameter `x' (or the highest numbered `x_' parameter) must   */
/*   also be declared `INEXACT'.                                             */

#define Fast_Two_Sum_Tail(a, b, x, y) \
  bvirt = x - a; \
  y = b - bvirt

#define Fast_Two_Sum(a, b, x, y) \
  x = (REAL) (a + b); \
  Fast_Two_Sum_Tail(a, b, x, y)

#define Two_Sum_Tail(a, b, x, y) \
  bvirt = (REAL) (x - a); \
  avirt = x - bvirt; \
  bround = b - bvirt; \
  around = a - avirt; \
  y = around + bround

#define Two_Sum(a, b, x, y) \
  x = (REAL) (a + b); \
  Two_Sum_Tail(a, b, x, y)

#define Two_Diff_Tail(a, b, x, y) \
  bvirt = (REAL) (a - x); \
  avirt = x + bvirt; \
  bround = bvirt - b; \
  around = a - avirt; \
  y = around + bround

#define Two_Diff(a, b, x, y) \
  x = (REAL) (a - b); \
  Two_Diff_Tail(a, b, x, y)

#define Split(a, ahi, alo) \
  c = MUL(predConsts[ Splitter ], a); \
  abig = (REAL) (c - a); \
  ahi = c - abig; \
  alo = a - ahi

#define Two_Product_Tail(a, b, x, y) \
  Split(a, ahi, alo); \
  Split(b, bhi, blo); \
  err1 = x - MUL(ahi, bhi); \
  err2 = err1 - MUL(alo, bhi); \
  err3 = err2 - MUL(ahi, blo); \
  y = MUL(alo, blo) - err3

#define Two_Product(a, b, x, y) \
  x = MUL(a, b); \
  Two_Product_Tail(a, b, x, y)

/* Two_Product_Presplit() is Two_Product() where one of the inputs has       */
/*   already been split.  Avoids redundant splitting.                        */

#define Two_Product_Presplit(a, b, bhi, blo, x, y) \
  x = MUL(a, b); \
  Split(a, ahi, alo); \
  err1 = x - MUL(ahi, bhi); \
  err2 = err1 - MUL(alo, bhi); \
  err3 = err2 - MUL(ahi, blo); \
  y = MUL(alo, blo) - err3

/* Macros for summing expansions of various fixed lengths.  These are all    */
/*   unrolled versions of Expansion_Sum().                                   */

#define Two_One_Diff(a1, a0, b, x2, x1, x0) \
  Two_Diff(a0, b , _i, x0); \
  Two_Sum( a1, _i, x2, x1)

#define Two_Two_Diff(a1, a0, b1, b0, x3, x2, x1, x0) \
  Two_One_Diff(a1, a0, b0, _j, _0, x0); \
  Two_One_Diff(_j, _0, b1, x3, x2, x1)

/* Macros for multiplying expansions of various fixed lengths.               */

#define Two_One_Product(a1, a0, b, x3, x2, x1, x0) \
  Split(b, bhi, blo); \
  Two_Product_Presplit(a0, b, bhi, blo, _i, x0); \
  Two_Product_Presplit(a1, b, bhi, blo, _j, _0); \
  Two_Sum(_i, _0, _k, x1); \
  Fast_Two_Sum(_j, _k, x3, x2)

/*****************************************************************************/
/*                                                                           */
/*  exactinit()   Initialize the variables used for exact arithmetic.        */
/*                                                                           */
/*  `epsilon' is the largest power of two such that 1.0 + epsilon = 1.0 in   */
/*  floating-point arithmetic.  `epsilon' bounds the relative roundoff       */
/*  error.  It is used for floating-point error analysis.                    */
/*                                                                           */
/*  `splitter' is used to split floating-point numbers into two half-        */
/*  length significands for exact multiplication.                            */
/*                                                                           */
/*  I imagine that a highly optimizing compiler might be too smart for its   */
/*  own good, and somehow cause this routine to fail, if it pretends that    */
/*  floating-point arithmetic is too much like real arithmetic.              */
/*                                                                           */
/*  Don't change this routine unless you fully understand it.                */
/*                                                                           */
/*****************************************************************************/

__global__ void kerInitPredicate( REAL* predConsts )
{
    REAL half;
    REAL epsilon, splitter; 
    REAL check, lastcheck;
    int every_other;

    every_other = 1;
    half        = 0.5;
    epsilon     = 1.0;
    splitter    = 1.0;
    check       = 1.0;

    /* Repeatedly divide `epsilon' by two until it is too small to add to    */
    /*   one without causing roundoff.  (Also check if the sum is equal to   */
    /*   the previous sum, for machines that round up instead of using exact */
    /*   rounding.  Not that this library will work on such machines anyway. */
    do
    {
        lastcheck   = check;
        epsilon     *= half;

        if (every_other)
        {
            splitter *= 2.0;
        }

        every_other = !every_other;
        check       = 1.0 + epsilon;
    } while ((check != 1.0) && (check != lastcheck));

    /* Error bounds for orientation and incircle tests. */
    predConsts[ Epsilon ]           = epsilon; 
    predConsts[ Splitter ]          = splitter + 1.0;
    predConsts[ Resulterrbound ]    = (3.0 + 8.0 * epsilon) * epsilon;
    predConsts[ CcwerrboundA ]      = (3.0 + 16.0 * epsilon) * epsilon;
    predConsts[ CcwerrboundB ]      = (2.0 + 12.0 * epsilon) * epsilon;
    predConsts[ CcwerrboundC ]      = (9.0 + 64.0 * epsilon) * epsilon * epsilon;
    predConsts[ O3derrboundA ]      = (7.0 + 56.0 * epsilon) * epsilon;
    predConsts[ O3derrboundB ]      = (3.0 + 28.0 * epsilon) * epsilon;
    predConsts[ O3derrboundC ]      = (26.0 + 288.0 * epsilon) * epsilon * epsilon;
    predConsts[ IccerrboundA ]      = (10.0 + 96.0 * epsilon) * epsilon;
    predConsts[ IccerrboundB ]      = (4.0 + 48.0 * epsilon) * epsilon;
    predConsts[ IccerrboundC ]      = (44.0 + 576.0 * epsilon) * epsilon * epsilon;
    predConsts[ IsperrboundA ]      = (16.0 + 224.0 * epsilon) * epsilon;
    predConsts[ IsperrboundB ]      = (5.0 + 72.0 * epsilon) * epsilon;
    predConsts[ IsperrboundC ]      = (71.0 + 1408.0 * epsilon) * epsilon * epsilon;

    return;
}

/*****************************************************************************/
/*                                                                           */
/*  scale_expansion_zeroelim()   Multiply an expansion by a scalar,          */
/*                               eliminating zero components from the        */
/*                               output expansion.                           */
/*                                                                           */
/*  Sets h = be.  See either version of my paper for details.                */
/*                                                                           */
/*  Maintains the nonoverlapping property.  If round-to-even is used (as     */
/*  with IEEE 754), maintains the strongly nonoverlapping and nonadjacent    */
/*  properties as well.  (That is, if e has one of these properties, so      */
/*  will h.)                                                                 */
/*                                                                           */
/*****************************************************************************/

/* e and h cannot be the same. */
__device__ int scale_expansion_zeroelim
(
const REAL* predConsts,
int         elen,
REAL*       e,
REAL        b,
REAL*       h
)
{
  INEXACT REAL Q, sum;
  REAL hh;
  INEXACT REAL product1;
  REAL product0;
  int eindex, hindex;
  REAL enow;
  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;

  Split(b, bhi, blo);
  Two_Product_Presplit(e[0], b, bhi, blo, Q, hh);
  hindex = 0;
  if (hh != 0) {
    h[hindex++] = hh;
  }
  for (eindex = 1; eindex < elen; eindex++) {
    enow = e[eindex];
    Two_Product_Presplit(enow, b, bhi, blo, product1, product0);
    Two_Sum(Q, product0, sum, hh);
    if (hh != 0) {
      h[hindex++] = hh;
    }
    Fast_Two_Sum(product1, sum, Q, hh);
    if (hh != 0) {
      h[hindex++] = hh;
    }
  }
  if ((Q != 0.0) || (hindex == 0)) {
    h[hindex++] = Q;
  }
  return hindex;
}

/*****************************************************************************/
/*                                                                           */
/*  fast_expansion_sum_zeroelim()   Sum two expansions, eliminating zero     */
/*                                  components from the output expansion.    */
/*                                                                           */
/*  Sets h = e + f.  See the long version of my paper for details.           */
/*                                                                           */
/*  If round-to-even is used (as with IEEE 754), maintains the strongly      */
/*  nonoverlapping property.  (That is, if e is strongly nonoverlapping, h   */
/*  will be also.)  Does NOT maintain the nonoverlapping or nonadjacent      */
/*  properties.                                                              */
/*                                                                           */
/*****************************************************************************/

/* h cannot be e or f. */
__device__ int _fast_expansion_sum_zeroelim
(
int elen,
REAL *e,
int flen,
REAL *f,
REAL *h
)
{
  REAL Q;
  INEXACT REAL Qnew;
  INEXACT REAL hh;
  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  int eindex, findex, hindex;
  REAL enow, fnow;

  enow = e[0];
  fnow = f[0];
  eindex = findex = 0;
  if ((fnow > enow) == (fnow > -enow)) {
    Q = enow;
    enow = e[++eindex];
  } else {
    Q = fnow;
    fnow = f[++findex];
  }
  hindex = 0;
  if ((eindex < elen) && (findex < flen)) {
    if ((fnow > enow) == (fnow > -enow)) {
      Fast_Two_Sum(enow, Q, Qnew, hh);
      enow = e[++eindex];
    } else {
      Fast_Two_Sum(fnow, Q, Qnew, hh);
      fnow = f[++findex];
    }
    Q = Qnew;
    if (hh != 0.0) {
      h[hindex++] = hh;
    }
    while ((eindex < elen) && (findex < flen)) {
      if ((fnow > enow) == (fnow > -enow)) {
        Two_Sum(Q, enow, Qnew, hh);
        enow = e[++eindex];
      } else {
        Two_Sum(Q, fnow, Qnew, hh);
        fnow = f[++findex];
      }
      Q = Qnew;
      if (hh != 0.0) {
        h[hindex++] = hh;
      }
    }
  }
  while (eindex < elen) {
    Two_Sum(Q, enow, Qnew, hh);
    enow = e[++eindex];
    Q = Qnew;
    if (hh != 0.0) {
      h[hindex++] = hh;
    }
  }
  while (findex < flen) {
    Two_Sum(Q, fnow, Qnew, hh);
    fnow = f[++findex];
    Q = Qnew;
    if (hh != 0.0) {
      h[hindex++] = hh;
    }
  }
  if ((Q != 0.0) || (hindex == 0)) {
    h[hindex++] = Q;
  }
  return hindex;
}

__device__ int squared_scale_expansion_zeroelim
(
 const REAL* predConsts,
 int elen,
 REAL *e,
 REAL b,
 REAL *h
 )
{
    INEXACT REAL Q, sum, Q2, sum2;
    REAL hh;
    INEXACT REAL product1, product2;
    REAL product0;
    int eindex, hindex;
    REAL enow;
    INEXACT REAL bvirt;
    REAL avirt, bround, around;
    INEXACT REAL c;
    INEXACT REAL abig;
    REAL ahi, alo, bhi, blo;
    REAL err1, err2, err3;

    hindex = 0;

    Split(b, bhi, blo);
    Two_Product_Presplit(e[0], b, bhi, blo, Q, hh);
    Two_Product_Presplit(hh, b, bhi, blo, Q2, hh); 

    if (hh != 0) {
        h[hindex++] = hh; 
    }

    for (eindex = 1; eindex < elen; eindex++) {
        enow = e[eindex];
        Two_Product_Presplit(enow, b, bhi, blo, product1, product0);
        Two_Sum(Q, product0, sum, hh);

        Two_Product_Presplit(hh, b, bhi, blo, product2, product0); 
        Two_Sum(Q2, product0, sum2, hh); 
        if (hh != 0) {
            h[hindex++] = hh; 
        }

        Fast_Two_Sum(product2, sum2, Q2, hh); 
        if (hh != 0) {
            h[hindex++] = hh; 
        }

        Fast_Two_Sum(product1, sum, Q, hh);

        Two_Product_Presplit(hh, b, bhi, blo, product2, product0); 
        Two_Sum(Q2, product0, sum2, hh);
        if (hh != 0) {
            h[hindex++] = hh; 
        }

        Fast_Two_Sum(product2, sum2, Q2, hh); 
        if (hh != 0) {
            h[hindex++] = hh; 
        }
    }

    if (Q != 0) {
        Two_Product_Presplit(Q, b, bhi, blo, product2, product0); 
        Two_Sum(Q2, product0, sum2, hh);

        if (hh != 0) {
            h[hindex++] = hh; 
        }

        Fast_Two_Sum(product2, sum2, Q2, hh); 
        if (hh != 0) {
            h[hindex++] = hh; 
        }
    }

    if ((Q2 != 0) || (hindex == 0)) {
        h[hindex++] = Q2; 
    }

    return hindex;
}

/*****************************************************************************/
/*                                                                           */
/*  orient2dexact()   Exact 2D orientation test.  Robust.                    */
/*               Return a positive value if the points pa, pb, and pc occur  */
/*               in counterclockwise order; a negative value if they occur   */
/*               in clockwise order; and zero if they are collinear.  The    */
/*               result is also a rough approximation of twice the signed    */
/*               area of the triangle defined by the three points.           */
/*                                                                           */
/*  The function uses exact arithmetic to ensure a correct answer.  The      */
/*  result returned is the determinant of a matrix.                          */
/*                                                                           */
/*****************************************************************************/

__device__ REAL orient2dexact
(
const REAL* predConsts,
const REAL* pa,
const REAL* pb,
const REAL* pc
)
{
  INEXACT REAL axby1, axcy1, bxcy1, bxay1, cxay1, cxby1;
  REAL axby0, axcy0, bxcy0, bxay0, cxay0, cxby0;
  REAL aterms[4], bterms[4], cterms[4];
  INEXACT REAL aterms3, bterms3, cterms3;
  REAL v[8], w[12];
  int vlength, wlength;

  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;
  INEXACT REAL _i, _j;
  REAL _0;

  Two_Product(pa[0], pb[1], axby1, axby0);
  Two_Product(pa[0], pc[1], axcy1, axcy0);
  Two_Two_Diff(axby1, axby0, axcy1, axcy0,
               aterms3, aterms[2], aterms[1], aterms[0]);
  aterms[3] = aterms3;

  Two_Product(pb[0], pc[1], bxcy1, bxcy0);
  Two_Product(pb[0], pa[1], bxay1, bxay0);
  Two_Two_Diff(bxcy1, bxcy0, bxay1, bxay0,
               bterms3, bterms[2], bterms[1], bterms[0]);
  bterms[3] = bterms3;

  Two_Product(pc[0], pa[1], cxay1, cxay0);
  Two_Product(pc[0], pb[1], cxby1, cxby0);
  Two_Two_Diff(cxay1, cxay0, cxby1, cxby0,
               cterms3, cterms[2], cterms[1], cterms[0]);
  cterms[3] = cterms3;

  vlength = _fast_expansion_sum_zeroelim(4, aterms, 4, bterms, v);
  wlength = _fast_expansion_sum_zeroelim(vlength, v, 4, cterms, w);

  return w[wlength - 1];
}

/*****************************************************************************/
/*                                                                           */
/*  orient3dexact()   Exact 3D orientation test.  Robust.                    */
/*  orient3dfast()    Do a fast check, return 0 if unsure                    */
/*                                                                           */
/*               Return a positive value if the point pd lies below the      */
/*               plane passing through pa, pb, and pc; "below" is defined so */
/*               that pa, pb, and pc appear in counterclockwise order when   */
/*               viewed from above the plane.  Returns a negative value if   */
/*               pd lies above the plane.  Returns zero if the points are    */
/*               coplanar.  The result is also a rough approximation of six  */
/*               times the signed volume of the tetrahedron defined by the   */
/*               four points.                                                */
/*                                                                           */
/*  The function uses exact arithmetic to ensure a correct answer.  The      */
/*  result returned is the determinant of a matrix.                          */
/*                                                                           */
/*****************************************************************************/

__device__ REAL orient3dexact
(
const REAL* predConsts,
const REAL* pa,
const REAL* pb,
const REAL* pc,
const REAL* pd
)
{
  INEXACT REAL axby1, bxcy1, cxdy1, dxay1, axcy1, bxdy1;
  INEXACT REAL bxay1, cxby1, dxcy1, axdy1, cxay1, dxby1;
  REAL axby0, bxcy0, cxdy0, dxay0, axcy0, bxdy0;
  REAL bxay0, cxby0, dxcy0, axdy0, cxay0, dxby0;
  REAL ab[4], bc[4], cd[4], da[4], ac[4], bd[4];
  REAL temp8[8];
  int templen;
  REAL abc[12], bcd[12], cda[12], dab[12];
  int abclen, bcdlen, cdalen, dablen;
  REAL adet[24], bdet[24], cdet[24], ddet[24];
  int alen, blen, clen, dlen;
  REAL abdet[48], cddet[48];
  int ablen, cdlen;
  REAL deter[96];
  int deterlen;
  int i;

  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;
  INEXACT REAL _i, _j;
  REAL _0;

  Two_Product(pa[0], pb[1], axby1, axby0);
  Two_Product(pb[0], pa[1], bxay1, bxay0);
  Two_Two_Diff(axby1, axby0, bxay1, bxay0, ab[3], ab[2], ab[1], ab[0]);

  Two_Product(pb[0], pc[1], bxcy1, bxcy0);
  Two_Product(pc[0], pb[1], cxby1, cxby0);
  Two_Two_Diff(bxcy1, bxcy0, cxby1, cxby0, bc[3], bc[2], bc[1], bc[0]);

  Two_Product(pc[0], pd[1], cxdy1, cxdy0);
  Two_Product(pd[0], pc[1], dxcy1, dxcy0);
  Two_Two_Diff(cxdy1, cxdy0, dxcy1, dxcy0, cd[3], cd[2], cd[1], cd[0]);

  Two_Product(pd[0], pa[1], dxay1, dxay0);
  Two_Product(pa[0], pd[1], axdy1, axdy0);
  Two_Two_Diff(dxay1, dxay0, axdy1, axdy0, da[3], da[2], da[1], da[0]);

  Two_Product(pa[0], pc[1], axcy1, axcy0);
  Two_Product(pc[0], pa[1], cxay1, cxay0);
  Two_Two_Diff(axcy1, axcy0, cxay1, cxay0, ac[3], ac[2], ac[1], ac[0]);

  Two_Product(pb[0], pd[1], bxdy1, bxdy0);
  Two_Product(pd[0], pb[1], dxby1, dxby0);
  Two_Two_Diff(bxdy1, bxdy0, dxby1, dxby0, bd[3], bd[2], bd[1], bd[0]);

  templen = _fast_expansion_sum_zeroelim(4, cd, 4, da, temp8);
  cdalen = _fast_expansion_sum_zeroelim(templen, temp8, 4, ac, cda);
  templen = _fast_expansion_sum_zeroelim(4, da, 4, ab, temp8);
  dablen = _fast_expansion_sum_zeroelim(templen, temp8, 4, bd, dab);
  for (i = 0; i < 4; i++) {
    bd[i] = -bd[i];
    ac[i] = -ac[i];
  }
  templen = _fast_expansion_sum_zeroelim(4, ab, 4, bc, temp8);
  abclen = _fast_expansion_sum_zeroelim(templen, temp8, 4, ac, abc);
  templen = _fast_expansion_sum_zeroelim(4, bc, 4, cd, temp8);
  bcdlen = _fast_expansion_sum_zeroelim(templen, temp8, 4, bd, bcd);

  alen = scale_expansion_zeroelim(predConsts, bcdlen, bcd, pa[2], adet);
  blen = scale_expansion_zeroelim(predConsts, cdalen, cda, -pb[2], bdet);
  clen = scale_expansion_zeroelim(predConsts, dablen, dab, pc[2], cdet);
  dlen = scale_expansion_zeroelim(predConsts, abclen, abc, -pd[2], ddet);

  ablen = _fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
  cdlen = _fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet);
  deterlen = _fast_expansion_sum_zeroelim(ablen, abdet, cdlen, cddet, deter);

  return deter[deterlen - 1];
}

__device__ REAL orient3dfast
(
const REAL* predConsts,
const REAL* pa,
const REAL* pb,
const REAL* pc,
const REAL* pd
)
{
  REAL adx, bdx, cdx, ady, bdy, cdy, adz, bdz, cdz;
  REAL bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
  REAL det;
  REAL permanent, errbound;

  adx = pa[0] - pd[0];
  bdx = pb[0] - pd[0];
  cdx = pc[0] - pd[0];
  ady = pa[1] - pd[1];
  bdy = pb[1] - pd[1];
  cdy = pc[1] - pd[1];
  adz = pa[2] - pd[2];
  bdz = pb[2] - pd[2];
  cdz = pc[2] - pd[2];

  bdxcdy = bdx * cdy;
  cdxbdy = cdx * bdy;

  cdxady = cdx * ady;
  adxcdy = adx * cdy;

  adxbdy = adx * bdy;
  bdxady = bdx * ady;

  det = adz * (bdxcdy - cdxbdy) 
      + bdz * (cdxady - adxcdy)
      + cdz * (adxbdy - bdxady);

  permanent = (Absolute(bdxcdy) + Absolute(cdxbdy)) * Absolute(adz)
            + (Absolute(cdxady) + Absolute(adxcdy)) * Absolute(bdz)
            + (Absolute(adxbdy) + Absolute(bdxady)) * Absolute(cdz);
  errbound = predConsts[ O3derrboundA ] * permanent;
  if ((det > errbound) || (-det > errbound)) {
    return det;
  }

  return 0.0;
}

/*****************************************************************************/
/*                                                                           */
/*  inspherefast()   Approximate 3D insphere test.  Nonrobust.               */
/*  insphereexact()   Exact 3D insphere test.  Robust.                       */
/*  insphereslow()   Another exact 3D insphere test.  Robust.                */
/*  insphere()   Adaptive exact 3D insphere test.  Robust.                   */
/*                                                                           */
/*               Return a positive value if the point pe lies inside the     */
/*               sphere passing through pa, pb, pc, and pd; a negative value */
/*               if it lies outside; and zero if the five points are         */
/*               cospherical.  The points pa, pb, pc, and pd must be ordered */
/*               so that they have a positive orientation (as defined by     */
/*               orient3d()), or the sign of the result will be reversed.    */
/*                                                                           */
/*  Only the first and last routine should be used; the middle two are for   */
/*  timings.                                                                 */
/*                                                                           */
/*  The last three use exact arithmetic to ensure a correct answer.  The     */
/*  result returned is the determinant of a matrix.  In insphere() only,     */
/*  this determinant is computed adaptively, in the sense that exact         */
/*  arithmetic is used only to the degree it is needed to ensure that the    */
/*  returned value has the correct sign.  Hence, insphere() is usually quite */
/*  fast, but will run more slowly when the input points are cospherical or  */
/*  nearly so.                                                               */
/*                                                                           */
/*****************************************************************************/

__noinline__ __device__ int mult_add
(
 const REAL     *predConsts, 
 REAL           *a,  
 int            lena, 
 REAL           fa,
 REAL           *b, 
 int            lenb, 
 REAL           fb,
 REAL           *c, 
 int            lenc, 
 REAL           fc, 
 REAL           *tempa, 
 REAL           *tempb, 
 REAL           *tempab, 
 REAL           *ret
) 
{
  int tempalen = scale_expansion_zeroelim(predConsts, lena, a, fa, tempa);
  int tempblen = scale_expansion_zeroelim(predConsts, lenb, b, fb, tempb);
  int tempablen = _fast_expansion_sum_zeroelim(tempalen, tempa, tempblen, tempb, tempab); 
  tempalen = scale_expansion_zeroelim(predConsts, lenc, c, fc, tempa);
  return _fast_expansion_sum_zeroelim(tempalen, tempa, tempablen, tempab, ret);
}

__noinline__ __device__ int calc_det
(
 const REAL     *predConsts, 
 REAL           *a,
 int            alen,
 REAL           *b,
 int            blen, 
 REAL           *c,
 int            clen,
 REAL           *d,
 int            dlen, 
 const REAL     *f,
 REAL           *temp1a, 
 REAL           *temp1b,
 REAL           *temp2,
 REAL           *temp8x,
 REAL           *temp8y,
 REAL           *temp8z,
 REAL           *temp16, 
 REAL           *ret24
)
{
  int temp1alen = _fast_expansion_sum_zeroelim(alen, a, blen, b, temp1a);
  int temp1blen = _fast_expansion_sum_zeroelim(clen, c, dlen, d, temp1b);
  for (int i = 0; i < temp1blen; i++) {
    temp1b[i] = -temp1b[i];
  }
  int temp2len = _fast_expansion_sum_zeroelim(temp1alen, temp1a, temp1blen, temp1b, temp2);
  int xlen = squared_scale_expansion_zeroelim(predConsts, temp2len, temp2, f[0], temp8x);
  int ylen = squared_scale_expansion_zeroelim(predConsts, temp2len, temp2, f[1], temp8y);
  int zlen = squared_scale_expansion_zeroelim(predConsts, temp2len, temp2, f[2], temp8z);
  int len = _fast_expansion_sum_zeroelim(xlen, temp8x, ylen, temp8y, temp16);

  return _fast_expansion_sum_zeroelim(len, temp16, zlen, temp8z, ret24);
}

__device__ REAL insphereexact
(
PredicateInfo   predInfo,
const REAL*     pa,
const REAL*     pb,
const REAL*     pc,
const REAL*     pd,
const REAL*     pe
)
{
    const REAL* predConsts  = predInfo._consts;
    REAL* predData          = predInfo._data;

    // Index into global memory
    REAL* temp96    = predData; predData += Temp96Size;     //  abcd, bcde, cdea, deab, eabc;
    REAL* det384x   = predData; predData += Det384xSize;
    REAL* det384y   = predData; predData += Det384ySize;
    REAL* det384z   = predData; predData += Det384zSize;
    REAL* detxy     = predData; predData += DetxySize;
    REAL* adet      = predData; predData += AdetSize;
    REAL* abdet     = predData; predData += AbdetSize;
    REAL* cdedet    = predData; predData += CdedetSize;
    REAL* deter     = predData;
    REAL* bdet      = cdedet;
    REAL* cdet      = adet;
    REAL* ddet      = cdedet;
    REAL* edet      = adet;
    REAL* cddet     = deter;

  INEXACT REAL axby1, bxcy1, cxdy1, dxey1, exay1;
  INEXACT REAL bxay1, cxby1, dxcy1, exdy1, axey1;
  INEXACT REAL axcy1, bxdy1, cxey1, dxay1, exby1;
  INEXACT REAL cxay1, dxby1, excy1, axdy1, bxey1;
  REAL axby0, bxcy0, cxdy0, dxey0, exay0;
  REAL bxay0, cxby0, dxcy0, exdy0, axey0;
  REAL axcy0, bxdy0, cxey0, dxay0, exby0;
  REAL cxay0, dxby0, excy0, axdy0, bxey0;
  REAL ab[4], bc[4], cd[4], de[4], ea[4];
  REAL ac[4], bd[4], ce[4], da[4], eb[4];
  REAL temp8a[8], temp8b[8], temp16[16];
  REAL abc[24], bcd[24], cde[24], dea[24], eab[24];
  REAL abd[24], bce[24], cda[24], deb[24], eac[24];
  int abclen, bcdlen, cdelen, dealen, eablen;
  int abdlen, bcelen, cdalen, deblen, eaclen;
  REAL temp48a[48], temp48b[48];
  int alen, blen, clen, dlen, elen;
  int ablen, cdlen;
  int deterlen;

  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;
  INEXACT REAL _i, _j;
  REAL _0;

  // (pa[0] * pb[1]) # (pb[0] * pa[1]) => ab[0..3]
  Two_Product(pa[0], pb[1], axby1, axby0);
  Two_Product(pb[0], pa[1], bxay1, bxay0);
  Two_Two_Diff(axby1, axby0, bxay1, bxay0, ab[3], ab[2], ab[1], ab[0]);

  // => bc[0..3]
  Two_Product(pb[0], pc[1], bxcy1, bxcy0);
  Two_Product(pc[0], pb[1], cxby1, cxby0);
  Two_Two_Diff(bxcy1, bxcy0, cxby1, cxby0, bc[3], bc[2], bc[1], bc[0]);

  // => cd[0..3]
  Two_Product(pc[0], pd[1], cxdy1, cxdy0);
  Two_Product(pd[0], pc[1], dxcy1, dxcy0);
  Two_Two_Diff(cxdy1, cxdy0, dxcy1, dxcy0, cd[3], cd[2], cd[1], cd[0]);

  // => de[0..3]
  Two_Product(pd[0], pe[1], dxey1, dxey0);
  Two_Product(pe[0], pd[1], exdy1, exdy0);
  Two_Two_Diff(dxey1, dxey0, exdy1, exdy0, de[3], de[2], de[1], de[0]);

  // => ea[0..3]
  Two_Product(pe[0], pa[1], exay1, exay0);
  Two_Product(pa[0], pe[1], axey1, axey0);
  Two_Two_Diff(exay1, exay0, axey1, axey0, ea[3], ea[2], ea[1], ea[0]);

  // => ac[0..3]
  Two_Product(pa[0], pc[1], axcy1, axcy0);
  Two_Product(pc[0], pa[1], cxay1, cxay0);
  Two_Two_Diff(axcy1, axcy0, cxay1, cxay0, ac[3], ac[2], ac[1], ac[0]);

  // => bd[0..3]
  Two_Product(pb[0], pd[1], bxdy1, bxdy0);
  Two_Product(pd[0], pb[1], dxby1, dxby0);
  Two_Two_Diff(bxdy1, bxdy0, dxby1, dxby0, bd[3], bd[2], bd[1], bd[0]);

  // => ce[0..3]
  Two_Product(pc[0], pe[1], cxey1, cxey0);
  Two_Product(pe[0], pc[1], excy1, excy0);
  Two_Two_Diff(cxey1, cxey0, excy1, excy0, ce[3], ce[2], ce[1], ce[0]);

  // => da[0..3]
  Two_Product(pd[0], pa[1], dxay1, dxay0);
  Two_Product(pa[0], pd[1], axdy1, axdy0);
  Two_Two_Diff(dxay1, dxay0, axdy1, axdy0, da[3], da[2], da[1], da[0]);

  // => eb[0..3]
  Two_Product(pe[0], pb[1], exby1, exby0);
  Two_Product(pb[0], pe[1], bxey1, bxey0);
  Two_Two_Diff(exby1, exby0, bxey1, bxey0, eb[3], eb[2], eb[1], eb[0]);

  // pa[2] # pb[2] # pc[2] => abc[24]
  abclen = mult_add( predConsts, bc, 4, pa[2], ac, 4, -pb[2], ab, 4, pc[2], temp8a, temp8b, temp16, abc ); 

  // => bcd
  bcdlen = mult_add( predConsts, cd, 4, pb[2], bd, 4, -pc[2], bc, 4, pd[2], temp8a, temp8b, temp16, bcd ); 

  // => cde
  cdelen = mult_add( predConsts, de, 4, pc[2], ce, 4, -pd[2], cd, 4, pe[2], temp8a, temp8b, temp16, cde ); 

  // => dea
  dealen = mult_add( predConsts, ea, 4, pd[2], da, 4, -pe[2], de, 4, pa[2], temp8a, temp8b, temp16, dea ); 

  // => eab
  eablen = mult_add( predConsts, ab, 4, pe[2], eb, 4, -pa[2], ea, 4, pb[2], temp8a, temp8b, temp16, eab ); 

  // => abd
  abdlen = mult_add( predConsts, bd, 4, pa[2], da, 4, pb[2], ab, 4, pd[2], temp8a, temp8b, temp16, abd ); 

  // => bce
  bcelen = mult_add( predConsts, ce, 4, pb[2], eb, 4, pc[2], bc, 4, pe[2], temp8a, temp8b, temp16, bce ); 

  // => cda
  cdalen = mult_add( predConsts, da, 4, pc[2], ac, 4, pd[2], cd, 4, pa[2], temp8a, temp8b, temp16, cda ); 

  // => deb
  deblen = mult_add( predConsts, eb, 4, pd[2], bd, 4, pe[2], de, 4, pb[2], temp8a, temp8b, temp16, deb ); 

  // => eac
  eaclen = mult_add( predConsts, ac, 4, pe[2], ce, 4, pa[2], ea, 4, pc[2], temp8a, temp8b, temp16, eac ); 

  // => bcde
  alen = calc_det( predConsts, cde, cdelen, bce, bcelen, deb, deblen, bcd, bcdlen, pa, 
      temp48a, temp48b, temp96, det384x, det384y, det384z, detxy, adet ); 

  // => cdea
  blen = calc_det( predConsts, dea, dealen, cda, cdalen, eac, eaclen, cde, cdelen, pb, 
      temp48a, temp48b, temp96, det384x, det384y, det384z, detxy, bdet ); 

  ablen     = _fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);

  // => deab
  clen = calc_det( predConsts, eab, eablen, deb, deblen, abd, abdlen, dea, dealen, pc, 
      temp48a, temp48b, temp96, det384x, det384y, det384z, detxy, cdet ); 

  // => eabc
  dlen = calc_det( predConsts, abc, abclen, eac, eaclen, bce, bcelen, eab, eablen, pd, 
      temp48a, temp48b, temp96, det384x, det384y, det384z, detxy, ddet ); 

  cdlen     = _fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet);

  // => abcd
  elen = calc_det( predConsts, bcd, bcdlen, abd, abdlen, cda, cdalen, abc, abclen, pe, 
      temp48a, temp48b, temp96, det384x, det384y, det384z, detxy, edet ); 

  cdelen    = _fast_expansion_sum_zeroelim(cdlen, cddet, elen, edet, cdedet);
  deterlen  = _fast_expansion_sum_zeroelim(ablen, abdet, cdelen, cdedet, deter);

  return deter[deterlen - 1];
}

__forceinline__ __device__ REAL insphere
(
PredicateInfo   predInfo,
const REAL*     pa,
const REAL*     pb,
const REAL*     pc,
const REAL*     pd,
const REAL*     pe
)
{
    REAL aex, bex, cex, dex;
    REAL aey, bey, cey, dey;
    REAL aez, bez, cez, dez;
    REAL aexbey, bexaey, bexcey, cexbey, cexdey, dexcey, dexaey, aexdey;
    REAL aexcey, cexaey, bexdey, dexbey;
    REAL alift, blift, clift, dlift;
    REAL ab, bc, cd, da, ac, bd;
    REAL abplus, bcplus, cdplus, daplus, acplus, bdplus;
    REAL aezplus, bezplus, cezplus, dezplus;
    REAL det;
    REAL permanent, errbound;

    aex = pa[0] - pe[0];
    bex = pb[0] - pe[0];
    cex = pc[0] - pe[0];
    dex = pd[0] - pe[0];
    aey = pa[1] - pe[1];
    bey = pb[1] - pe[1];
    cey = pc[1] - pe[1];
    dey = pd[1] - pe[1];
    aez = pa[2] - pe[2];
    bez = pb[2] - pe[2];
    cez = pc[2] - pe[2];
    dez = pd[2] - pe[2];

    alift = aex * aex + aey * aey + aez * aez;
    blift = bex * bex + bey * bey + bez * bez;
    clift = cex * cex + cey * cey + cez * cez;
    dlift = dex * dex + dey * dey + dez * dez;

    aexbey      = aex * bey;
    bexaey      = bex * aey;
    abplus      = Absolute(aexbey) + Absolute(bexaey); 
    ab          = aexbey - bexaey;

    bexcey      = bex * cey;
    cexbey      = cex * bey;
    bcplus      = Absolute(bexcey) + Absolute(cexbey);
    bc          = bexcey - cexbey;

    cexdey      = cex * dey;
    dexcey      = dex * cey;
    cdplus      = Absolute(cexdey) + Absolute(dexcey); 
    cd          = cexdey - dexcey;

    dexaey      = dex * aey;
    aexdey      = aex * dey;
    daplus      = Absolute(dexaey) + Absolute(aexdey); 
    da          = dexaey - aexdey;

    aexcey      = aex * cey;
    cexaey      = cex * aey;
    acplus      = Absolute(aexcey) + Absolute(cexaey);
    ac          = aexcey - cexaey;

    bexdey      = bex * dey;
    dexbey      = dex * bey;
    bdplus      = Absolute(bexdey) + Absolute(dexbey); 
    bd          = bexdey - dexbey;

    det = ( cd * blift - bd * clift + bc * dlift ) * aez
        + (-cd * alift - da * clift - ac * dlift ) * bez
        + ( bd * alift + da * blift + ab * dlift ) * cez
        + (-bc * alift + ac * blift - ab * clift ) * dez; 

    aezplus = Absolute(aez);
    bezplus = Absolute(bez);
    cezplus = Absolute(cez);
    dezplus = Absolute(dez);
    permanent = ( cdplus * blift + bdplus * clift + bcplus * dlift ) * aezplus 
            + ( cdplus * alift + daplus * clift + acplus * dlift ) * bezplus 
            + ( bdplus * alift + daplus * blift + abplus * dlift ) * cezplus
            + ( bcplus * alift + acplus * blift + abplus * clift ) * dezplus; 

    errbound = predInfo._consts[ IsperrboundA ] * permanent;

    if ((det > errbound) || (-det > errbound))
    {
        return det;
    }

    return 0;   // Needs exact predicate
}

// det  = ( pa[0]^2 + pa[1]^2 + pa[2]^2 ) - ( pb[0]^2 + pb[1]^2 + pc[1]^2 )
__device__ REAL orient1dexact_lifted
(
const REAL* predConsts,
const REAL* pa,
const REAL* pb, 
bool        lifted
)
{
  if (!lifted) 
      return (pa[0] - pb[0]); 

  REAL axax1, ayay1, azaz1, bxbx1, byby1, bzbz1;
  REAL axax0, ayay0, azaz0, bxbx0, byby0, bzbz0;
  REAL aterms[4], bterms[4], cterms[4];
  REAL aterms3, bterms3, cterms3;
  REAL v[8], w[12];
  int vlength, wlength;

  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;
  INEXACT REAL _i, _j;
  REAL _0;

  Two_Product(pa[0], pa[0], axax1, axax0);
  Two_Product(pb[0], pb[0], bxbx1, bxbx0);
  Two_Two_Diff(axax1, axax0, bxbx1, bxbx0,
               aterms3, aterms[2], aterms[1], aterms[0]);
  aterms[3] = aterms3;

  Two_Product(pa[1], pa[1], ayay1, ayay0);
  Two_Product(pb[1], pb[1], byby1, byby0);
  Two_Two_Diff(ayay1, ayay0, byby1, byby0,
               bterms3, bterms[2], bterms[1], bterms[0]);
  bterms[3] = bterms3;

  Two_Product(pa[2], pa[2], azaz1, azaz0);
  Two_Product(pb[2], pb[2], bzbz1, bzbz0);
  Two_Two_Diff(azaz1, azaz0, bzbz1, bzbz0,
               cterms3, cterms[2], cterms[1], cterms[0]);
  cterms[3] = cterms3;

  vlength = _fast_expansion_sum_zeroelim(4, aterms, 4, bterms, v);
  wlength = _fast_expansion_sum_zeroelim(vlength, v, 4, cterms, w);

  return w[wlength - 1];
}

__device__ REAL orient2dexact_lifted
(
PredicateInfo   predInfo,
const REAL*     pa,
const REAL*     pb,
const REAL*     pc,
bool            lifted
)
{
  const REAL* predConsts = predInfo._consts;
  REAL* predData         = predInfo._data;

  REAL* aterms = predData; predData += 24;
  REAL* bterms = predData; predData += 24;
  REAL* cterms = predData; predData += 24;
  REAL* v      = predData; predData += 48;
  REAL* w      = predData; predData += 72;

  REAL aax1, aax0, aay1, aay0, aaz[2]; 
  REAL temp[4]; 
  REAL palift[6], pblift[6], pclift[6]; 
  REAL xy1terms[12], xy2terms[12]; 

  int palen, pblen, pclen; 
  int xy1len, xy2len; 
  int alen, blen, clen; 
  int vlen, wlen; 

  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;
  INEXACT REAL _i, _j;
  REAL _0;

  // Compute the lifted coordinate
  if (lifted) 
  {
      Two_Product(pa[0], pa[0], aax1, aax0); 
      Two_Product(-pa[1], pa[1], aay1, aay0); 
      Two_Product(pa[2], pa[2], aaz[1], aaz[0]); 
      Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]); 
      palen = _fast_expansion_sum_zeroelim(4, temp, 2, aaz, palift);
      
      Two_Product(pb[0], pb[0], aax1, aax0); 
      Two_Product(-pb[1], pb[1], aay1, aay0); 
      Two_Product(pb[2], pb[2], aaz[1], aaz[0]); 
      Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]); 
      pblen = _fast_expansion_sum_zeroelim(4, temp, 2, aaz, pblift);
      
      Two_Product(pc[0], pc[0], aax1, aax0); 
      Two_Product(-pc[1], pc[1], aay1, aay0); 
      Two_Product(pc[2], pc[2], aaz[1], aaz[0]); 
      Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]); 
      pclen = _fast_expansion_sum_zeroelim(4, temp, 2, aaz, pclift);
  } else
  {
      palen = 1; palift[0] = pa[1]; 
      pblen = 1; pblift[0] = pb[1]; 
      pclen = 1; pclift[0] = pc[1]; 
  }

  // Compute the determinant as usual
  xy1len = scale_expansion_zeroelim(predConsts, pblen, pblift, pa[0], xy1terms);
  xy2len = scale_expansion_zeroelim(predConsts, pclen, pclift, -pa[0], xy2terms);
  alen = _fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, aterms);
  
  xy1len = scale_expansion_zeroelim(predConsts, pclen, pclift, pb[0], xy1terms);
  xy2len = scale_expansion_zeroelim(predConsts, palen, palift, -pb[0], xy2terms);
  blen = _fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, bterms);
  
  xy1len = scale_expansion_zeroelim(predConsts, palen, palift, pc[0], xy1terms);
  xy2len = scale_expansion_zeroelim(predConsts, pblen, pblift, -pc[0], xy2terms);
  clen = _fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, cterms);
  
  vlen = _fast_expansion_sum_zeroelim(alen, aterms, blen, bterms, v);
  wlen = _fast_expansion_sum_zeroelim(vlen, v, clen, cterms, w);

  return w[wlen - 1];
}

__device__ REAL orient3dexact_lifted
(
PredicateInfo   predInfo,
const REAL*     pa,
const REAL*     pb,
const REAL*     pc,
const REAL*     pd,
bool            lifted
)
{
    const REAL* predConsts = predInfo._consts;
    REAL* predData         = predInfo._data;

    // Index into global memory
    REAL* ab        = predData; predData += 24;
    REAL* bc        = predData; predData += 24;
    REAL* cd        = predData; predData += 24;
    REAL* da        = predData; predData += 24;
    REAL* ac        = predData; predData += 24;
    REAL* bd        = predData; predData += 24;
    REAL* temp48    = predData; predData += 48;
    REAL* cda       = predData; predData += 72;
    REAL* dab       = predData; predData += 72;
    REAL* abc       = predData; predData += 72;
    REAL* bcd       = predData; predData += 72;
    REAL* adet      = predData; predData += 144;
    REAL* bdet      = predData; predData += 144;
    REAL* cdet      = predData; predData += 144;
    REAL* ddet      = predData; predData += 144;
    REAL* abdet     = predData; predData += 288;
    REAL* cddet     = predData; predData += 288;
    REAL* deter     = predData; 

  REAL aax1, aax0, aay1, aay0, aaz[2]; 
  REAL temp[4]; 
  REAL palift[6], pblift[6], pclift[6], pdlift[6]; 
  REAL xy1terms[12], xy2terms[12]; 

  int templen; 
  int palen, pblen, pclen, pdlen; 
  int xy1len, xy2len; 
  int ablen, bclen, cdlen, dalen, aclen, bdlen; 
  int cdalen, dablen, abclen, bcdlen; 
  int alen, blen, clen, dlen; 
  int deterlen; 

  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;
  INEXACT REAL _i, _j;
  REAL _0;

  // Compute the lifted coordinate
  if (lifted) {
      Two_Product(pa[0], pa[0], aax1, aax0); 
      Two_Product(-pa[1], pa[1], aay1, aay0); 
      Two_Product(pa[2], pa[2], aaz[1], aaz[0]); 
      Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]); 
      palen = _fast_expansion_sum_zeroelim(4, temp, 2, aaz, palift);
      
      Two_Product(pb[0], pb[0], aax1, aax0); 
      Two_Product(-pb[1], pb[1], aay1, aay0); 
      Two_Product(pb[2], pb[2], aaz[1], aaz[0]); 
      Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]); 
      pblen = _fast_expansion_sum_zeroelim(4, temp, 2, aaz, pblift);
      
      Two_Product(pc[0], pc[0], aax1, aax0); 
      Two_Product(-pc[1], pc[1], aay1, aay0); 
      Two_Product(pc[2], pc[2], aaz[1], aaz[0]); 
      Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]); 
      pclen = _fast_expansion_sum_zeroelim(4, temp, 2, aaz, pclift);

      Two_Product(pd[0], pd[0], aax1, aax0); 
      Two_Product(-pd[1], pd[1], aay1, aay0); 
      Two_Product(pd[2], pd[2], aaz[1], aaz[0]); 
      Two_Two_Diff(aax1, aax0, aay1, aay0, temp[3], temp[2], temp[1], temp[0]); 
      pdlen = _fast_expansion_sum_zeroelim(4, temp, 2, aaz, pdlift);
  } else
  {
      palen = 1; palift[0] = pa[2]; 
      pblen = 1; pblift[0] = pb[2]; 
      pclen = 1; pclift[0] = pc[2]; 
      pdlen = 1; pdlift[0] = pd[2]; 
  }

  // Compute the determinant as usual
  xy1len = scale_expansion_zeroelim(predConsts, pblen, pblift, pa[1], xy1terms);
  xy2len = scale_expansion_zeroelim(predConsts, palen, palift, -pb[1], xy2terms);
  ablen = _fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, ab);
  
  xy1len = scale_expansion_zeroelim(predConsts, pclen, pclift, pb[1], xy1terms);
  xy2len = scale_expansion_zeroelim(predConsts, pblen, pblift, -pc[1], xy2terms);
  bclen = _fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, bc);

  xy1len = scale_expansion_zeroelim(predConsts, pdlen, pdlift, pc[1], xy1terms);
  xy2len = scale_expansion_zeroelim(predConsts, pclen, pclift, -pd[1], xy2terms);
  cdlen = _fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, cd);

  xy1len = scale_expansion_zeroelim(predConsts, palen, palift, pd[1], xy1terms);
  xy2len = scale_expansion_zeroelim(predConsts, pdlen, pdlift, -pa[1], xy2terms);
  dalen = _fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, da);

  xy1len = scale_expansion_zeroelim(predConsts, pclen, pclift, pa[1], xy1terms);
  xy2len = scale_expansion_zeroelim(predConsts, palen, palift, -pc[1], xy2terms);
  aclen = _fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, ac);

  xy1len = scale_expansion_zeroelim(predConsts, pdlen, pdlift, pb[1], xy1terms);
  xy2len = scale_expansion_zeroelim(predConsts, pblen, pblift, -pd[1], xy2terms);
  bdlen = _fast_expansion_sum_zeroelim(xy1len, xy1terms, xy2len, xy2terms, bd);

  templen = _fast_expansion_sum_zeroelim(cdlen, cd, dalen, da, temp48);
  cdalen = _fast_expansion_sum_zeroelim(templen, temp48, aclen, ac, cda);
  templen = _fast_expansion_sum_zeroelim(dalen, da, ablen, ab, temp48);
  dablen = _fast_expansion_sum_zeroelim(templen, temp48, bdlen, bd, dab);

  for (int i = 0; i < bdlen; i++) 
    bd[i] = -bd[i];
  for (int i = 0; i < aclen; i++) 
    ac[i] = -ac[i];
  
  templen = _fast_expansion_sum_zeroelim(ablen, ab, bclen, bc, temp48);
  abclen = _fast_expansion_sum_zeroelim(templen, temp48, aclen, ac, abc);
  templen = _fast_expansion_sum_zeroelim(bclen, bc, cdlen, cd, temp48);
  bcdlen = _fast_expansion_sum_zeroelim(templen, temp48, bdlen, bd, bcd);

  alen = scale_expansion_zeroelim(predConsts, bcdlen, bcd, pa[0], adet);
  blen = scale_expansion_zeroelim(predConsts, cdalen, cda, -pb[0], bdet);
  clen = scale_expansion_zeroelim(predConsts, dablen, dab, pc[0], cdet);
  dlen = scale_expansion_zeroelim(predConsts, abclen, abc, -pd[0], ddet);

  ablen = _fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
  cdlen = _fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet);
  deterlen = _fast_expansion_sum_zeroelim(ablen, abdet, cdlen, cddet, deter);

  return deter[deterlen - 1];
}

////////////////////////////////////////////////////////////////////////////////
