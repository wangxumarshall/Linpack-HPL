/* 
 * -- High Performance Computing Linpack Benchmark (HPL)                
 *    HPL - 2.3 - December 2, 2018                          
 *    Antoine P. Petitet                                                
 *    University of Tennessee, Knoxville                                
 *    Innovative Computing Laboratory                                 
 *    (C) Copyright 2000-2008 All Rights Reserved                       
 *                                                                      
 * -- Copyright notice and Licensing terms:                             
 *                                                                      
 * Redistribution  and  use in  source and binary forms, with or without
 * modification, are  permitted provided  that the following  conditions
 * are met:                                                             
 *                                                                      
 * 1. Redistributions  of  source  code  must retain the above copyright
 * notice, this list of conditions and the following disclaimer.        
 *                                                                      
 * 2. Redistributions in binary form must reproduce  the above copyright
 * notice, this list of conditions,  and the following disclaimer in the
 * documentation and/or other materials provided with the distribution. 
 *                                                                      
 * 3. All  advertising  materials  mentioning  features  or  use of this
 * software must display the following acknowledgement:                 
 * This  product  includes  software  developed  at  the  University  of
 * Tennessee, Knoxville, Innovative Computing Laboratory.             
 *                                                                      
 * 4. The name of the  University,  the name of the  Laboratory,  or the
 * names  of  its  contributors  may  not  be used to endorse or promote
 * products  derived   from   this  software  without  specific  written
 * permission.                                                          
 *                                                                      
 * -- Disclaimer:                                                       
 *                                                                      
 * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
 * ---------------------------------------------------------------------
 */ 
/*
 * Include files
 */
#include "hpl.h"

#ifdef STDC_HEADERS
void HPL_pdgesvK2
(
   HPL_T_grid *                     GRID,
   HPL_T_palg *                     ALGO,
   HPL_T_pmat *                     A
)
#else
void HPL_pdgesvK2
( GRID, ALGO, A )
   HPL_T_grid *                     GRID;
   HPL_T_palg *                     ALGO;
   HPL_T_pmat *                     A;
#endif
{
/* 
 * Purpose
 * =======
 *
 * HPL_pdgesvK2 factors a N+1-by-N matrix using LU factorization with row
 * partial pivoting.  The main algorithm  is the "right looking" variant
 * with look-ahead.  The  lower  triangular factor is left unpivoted and
 * the pivots are not returned. The right hand side is the N+1 column of
 * the coefficient matrix.
 *
 * Arguments
 * =========
 *
 * GRID    (local input)                 HPL_T_grid *
 *         On entry,  GRID  points  to the data structure containing the
 *         process grid information.
 *
 * ALGO    (global input)                HPL_T_palg *
 *         On entry,  ALGO  points to  the data structure containing the
 *         algorithmic parameters.
 *
 * A       (local input/output)          HPL_T_pmat *
 *         On entry, A points to the data structure containing the local
 *         array information.
 *
 * ---------------------------------------------------------------------
 */ 
/*
 * .. Local Variables ..
 */
   HPL_T_panel                * p, * * panel = NULL;
   HPL_T_UPD_FUN              HPL_pdupdate; 
   int                        N, depth, icurcol=0, j, jb, jj=0, jstart,
                              k, mycol, n, nb, nn, npcol, nq,
                              tag=MSGID_BEGIN_FACT, test=HPL_KEEP_TESTING;
#ifdef HPL_SDC_CHECK
   int                        myrank;
#endif
#ifdef HPL_PROGRESS_REPORT
   double start_time, time, gflops;
#endif
/* ..
 * .. Executable Statements ..
 */
   mycol = GRID->mycol; npcol        = GRID->npcol;
   depth = ALGO->depth; HPL_pdupdate = ALGO->upfun;
   N     = A->n;        nb           = A->nb;

   if( N <= 0 ) return;

#ifdef HPL_SDC_CHECK
   MPI_Comm_rank( GRID->all_comm, &myrank );
   HPL_sdc_log_init( &sdc_log_global, GRID->all_comm );
#endif

#ifdef HPL_PROGRESS_REPORT
   start_time = HPL_timer_walltime();
#endif

/*
 * Allocate a panel list of length depth + 1 (depth >= 1)
 */
   panel = (HPL_T_panel **)malloc( (size_t)(depth+1) * sizeof( HPL_T_panel *) );
   if( panel == NULL )
   { HPL_pabort( __LINE__, "HPL_pdgesvK2", "Memory allocation failed" ); }
/*
 * Create and initialize the first depth panels
 */
   nq = HPL_numroc( N+1, nb, nb, mycol, 0, npcol ); nn = N; jstart = 0;

   for( k = 0; k < depth; k++ )
   {
      jb = Mmin( nn, nb );
      HPL_pdpanel_new( GRID, ALGO, nn, nn+1, jb, A, jstart, jstart,
                       tag, &panel[k] );
      nn -= jb; jstart += jb;
      if( mycol == icurcol ) { jj += jb; nq -= jb; }
      icurcol = MModAdd1( icurcol, npcol );
      tag     = MNxtMgid( tag, MSGID_BEGIN_FACT, MSGID_END_FACT );
   }
/*
 * Create last depth+1 panel
 */
   HPL_pdpanel_new( GRID, ALGO, nn, nn+1, Mmin( nn, nb ), A, jstart,
                    jstart, tag, &panel[depth] );
   tag = MNxtMgid( tag, MSGID_BEGIN_FACT, MSGID_END_FACT );
/*
 * Initialize the lookahead - Factor jstart columns: panel[0..depth-1]
 */
   for( k = 0, j = 0; k < depth; k++ )
   {
      jb = jstart - j; jb = Mmin( jb, nb ); j += jb;
/*
 * Factor and broadcast k-th panel
 */
      HPL_pdfact(         panel[k] );
#ifdef HPL_SDC_CHECK
#if HPL_SDC_TRAIL_VERIFY
      /* Defensive pre-fill of CS_TRAIL after factorization */
      if( panel[k]->CS_TRAIL )
      {
         double * _Ai = ( panel[k]->grid->myrow == panel[k]->prow ?
                      Mptr( panel[k]->A, panel[k]->jb, 0, panel[k]->lda ) :
                      panel[k]->A );
         int _mi = ( panel[k]->grid->myrow == panel[k]->prow ?
                   panel[k]->mp - panel[k]->jb : panel[k]->mp );
         _mi = Mmax( 0, _mi );
         if( _mi > 0 && panel[k]->nq > 0 )
         {
            int _ji;
            for( _ji = 0; _ji < panel[k]->nq; _ji++ )
            {
               double _s = 0.0; int _ii;
               for( _ii = 0; _ii < _mi; _ii++ )
                  _s += ( panel[k]->CS_WEIGHTS ? panel[k]->CS_WEIGHTS[_ii] : 1.0 ) * _Ai[_ii + _ji * panel[k]->lda];
               panel[k]->CS_TRAIL[_ji] = _s;
            }
         }
      }
#endif
#endif
      (void) HPL_binit(   panel[k] );
#ifdef HPL_SDC_CHECK
      if( mycol == panel[k]->pcol && panel[k]->CS_PANEL )
      {
         int _ml2 = ( panel[k]->grid->myrow == panel[k]->prow ?
                      panel[k]->mp - panel[k]->jb : panel[k]->mp );
         _ml2 = Mmax( 0, _ml2 );
         HPL_sdc_compute_bcast_checksum( panel[k]->L2, panel[k]->ldl2,
            _ml2, panel[k]->L1, panel[k]->jb, panel[k]->DPIV,
            panel[k]->jb, panel[k]->CS_WEIGHTS, &(panel[k]->cs_bcast) );
      }
      else
      {
         panel[k]->cs_bcast = 0.0;
      }
      MPI_Bcast( &(panel[k]->cs_bcast), 1, MPI_DOUBLE,
                 panel[k]->pcol, GRID->row_comm );
#endif
      do
      { (void) HPL_bcast( panel[k], &test ); }
      while( test != HPL_SUCCESS );
      (void) HPL_bwait(   panel[k] );
#ifdef HPL_SDC_CHECK
      if( HPL_SDC_BCAST_VERIFY )
      {
         double cs_recv = 0.0;
         int _ml2 = ( panel[k]->grid->myrow == panel[k]->prow ?
                      panel[k]->mp - panel[k]->jb : panel[k]->mp );
         _ml2 = Mmax( 0, _ml2 );
         HPL_sdc_compute_bcast_checksum( panel[k]->L2, panel[k]->ldl2,
            _ml2, panel[k]->L1, panel[k]->jb, panel[k]->DPIV,
            panel[k]->jb, panel[k]->CS_WEIGHTS, &cs_recv );
         if( HPL_sdc_verify_checksum( panel[k]->cs_bcast, cs_recv,
                                      HPL_SDC_THRESHOLD ) )
         {
            double dev = panel[k]->cs_bcast - cs_recv;
            if( dev < 0.0 ) dev = -dev;
            HPL_sdc_log_fault( &sdc_log_global, myrank,
               panel[k]->grid->myrow, panel[k]->grid->mycol,
               HPL_SDC_FAULT_PANEL_BCAST, j,
               panel[k]->ia, panel[k]->ja,
               panel[k]->cs_bcast, cs_recv );
            HPL_pwarn( stdout, __LINE__, "HPL_pdgesvK2",
               "SDC detected in initial panel broadcast at column %d (k=%d) on rank %d (dev=%.3e)",
               j, k, myrank, dev );
         }
      }
#endif
/*
 * Partial update of the depth-k-1 panels in front of me
 */
      if( k < depth - 1 )
      {
         nn = HPL_numrocI( jstart-j, j, nb, nb, mycol, 0, npcol );
         HPL_pdupdate( NULL, NULL, panel[k], nn );
      }
   }
/*
 * Main loop over the remaining columns of A
 */
   for( j = jstart; j < N; j += nb )
   {
      n = N - j; jb = Mmin( n, nb );
#ifdef HPL_PROGRESS_REPORT
      /* if this is process 0,0 and not the first panel */
      if ( GRID->myrow == 0 && mycol == 0 && j > 0 ) 
      {
          time = HPL_timer_walltime() - start_time;
          gflops = 2.0*(N*(double)N*N - n*(double)n*n)/3.0/(time > 0.0 ? time : 1e-6)/1e9;
          HPL_fprintf( stdout, "Column=%09d Fraction=%4.1f%% Gflops=%9.3e\n", j, j*100.0/N, gflops);
      }
#endif
/*
 * Initialize current panel - Finish latest update, Factor and broadcast
 * current panel
 */
      (void) HPL_pdpanel_free( panel[depth] );
      HPL_pdpanel_init( GRID, ALGO, n, n+1, jb, A, j, j, tag, panel[depth] );

      if( mycol == icurcol )
      {
         nn = HPL_numrocI( jb, j, nb, nb, mycol, 0, npcol );
         for( k = 0; k < depth; k++ )   /* partial updates 0..depth-1 */
            (void) HPL_pdupdate( NULL, NULL, panel[k], nn );
         HPL_pdfact(       panel[depth] );    /* factor current panel */
#ifdef HPL_SDC_CHECK
#if HPL_SDC_TRAIL_VERIFY
         /* Defensive pre-fill of CS_TRAIL after factorization */
         if( panel[depth]->CS_TRAIL )
         {
            double * _Ai = ( panel[depth]->grid->myrow == panel[depth]->prow ?
                         Mptr( panel[depth]->A, panel[depth]->jb, 0, panel[depth]->lda ) :
                         panel[depth]->A );
            int _mi = ( panel[depth]->grid->myrow == panel[depth]->prow ?
                      panel[depth]->mp - panel[depth]->jb : panel[depth]->mp );
            _mi = Mmax( 0, _mi );
            if( _mi > 0 && panel[depth]->nq > 0 )
            {
               int _ji;
               for( _ji = 0; _ji < panel[depth]->nq; _ji++ )
               {
                  double _s = 0.0; int _ii;
                  for( _ii = 0; _ii < _mi; _ii++ )
                     _s += ( panel[depth]->CS_WEIGHTS ? panel[depth]->CS_WEIGHTS[_ii] : 1.0 ) * _Ai[_ii + _ji * panel[depth]->lda];
                  panel[depth]->CS_TRAIL[_ji] = _s;
               }
            }
         }
#endif
#endif
      }
      else { nn = 0; }
          /* binit must be called BEFORE cs_bcast computation: when HPL_COPY_L
           * is enabled, binit invokes HPL_copyL which fills L2 from A. */
      (void) HPL_binit( panel[depth] );
#ifdef HPL_SDC_CHECK
/*
 * Compute broadcast buffer checksum after binit (L2 is now filled).
 * Only owner column processes have valid L2 data; others set cs_bcast=0.
 * Use MPI_Bcast within each process row to propagate reference from icurcol.
 */
      if( mycol == icurcol && panel[depth]->CS_PANEL )
      {
         int _ml2 = ( panel[depth]->grid->myrow == panel[depth]->prow ?
                      panel[depth]->mp - panel[depth]->jb : panel[depth]->mp );
         _ml2 = Mmax( 0, _ml2 );
         HPL_sdc_compute_bcast_checksum( panel[depth]->L2, panel[depth]->ldl2,
            _ml2, panel[depth]->L1, panel[depth]->jb, panel[depth]->DPIV,
            panel[depth]->jb, panel[depth]->CS_WEIGHTS, &(panel[depth]->cs_bcast) );
      }
      else
      {
         panel[depth]->cs_bcast = 0.0;
      }
      MPI_Bcast( &(panel[depth]->cs_bcast), 1, MPI_DOUBLE,
                 icurcol, GRID->row_comm );
#ifdef HPL_SDC_INJECT
      {
         static int    _sdc_inj_col = -2;
         static double _sdc_inj_val = 1.0e6;
         if( _sdc_inj_col == -2 )
         {
            char * _ec = getenv("HPL_SDC_INJECT_COL");
            char * _ev = getenv("HPL_SDC_INJECT_VAL");
            _sdc_inj_col = _ec ? atoi(_ec) : (int)nb;
            _sdc_inj_val = _ev ? atof(_ev) : 1.0e6;
         }
         if( mycol == icurcol && j == _sdc_inj_col && panel[depth]->mp > 1 &&
             panel[depth]->L2 )
         {
            HPL_sdc_inject_at( panel[depth]->L2, 1, 0, _sdc_inj_val );
            HPL_pwarn( stdout, __LINE__, "HPL_pdgesvK2",
               "SDC FAULT INJECTED j=%d rank=%d (L2[1] replaced with %e)",
               j, myrank, _sdc_inj_val );
         }
      }
#endif
#endif
          /* Finish the latest update and broadcast the current panel */
      HPL_pdupdate( panel[depth], &test, panel[0], nq-nn );
      (void) HPL_bwait( panel[depth] );
#ifdef HPL_SDC_CHECK
      /* Verify broadcast integrity */
      if( HPL_SDC_BCAST_VERIFY )
      {
         double cs_recv = 0.0;
         int _ml2 = ( panel[depth]->grid->myrow == panel[depth]->prow ?
                      panel[depth]->mp - panel[depth]->jb : panel[depth]->mp );
         _ml2 = Mmax( 0, _ml2 );
         HPL_sdc_compute_bcast_checksum( panel[depth]->L2, panel[depth]->ldl2,
            _ml2, panel[depth]->L1, panel[depth]->jb, panel[depth]->DPIV,
            panel[depth]->jb, panel[depth]->CS_WEIGHTS, &cs_recv );
         if( HPL_sdc_verify_checksum( panel[depth]->cs_bcast, cs_recv,
                                      HPL_SDC_THRESHOLD ) )
         {
            double dev = fabs( panel[depth]->cs_bcast - cs_recv );
            HPL_sdc_log_fault( &sdc_log_global, myrank,
               panel[depth]->grid->myrow, panel[depth]->grid->mycol,
               HPL_SDC_FAULT_PANEL_BCAST, j,
               panel[depth]->ia, panel[depth]->ja,
               panel[depth]->cs_bcast, cs_recv );
            HPL_pwarn( stdout, __LINE__, "HPL_pdgesvK2",
               "SDC detected in panel broadcast at column %d on rank %d (dev=%.3e)",
               j, myrank, dev );
         }
      }
#endif
/*
 * Circular  of the panel pointers:
 * xtmp = x[0]; for( k=0; k < depth; k++ ) x[k] = x[k+1]; x[d] = xtmp;
 *
 * Go to next process row and column - update the message ids for broadcast
 */
      p = panel[0]; for( k = 0; k < depth; k++ ) panel[k] = panel[k+1];
      panel[depth] = p;

      if( mycol == icurcol ) { jj += jb; nq -= jb; }
      icurcol = MModAdd1( icurcol, npcol );
      tag     = MNxtMgid( tag, MSGID_BEGIN_FACT, MSGID_END_FACT );
   }
/*
 * Clean-up: Finish updates - release panels and panel list
 */
   nn = HPL_numrocI( 1, N, nb, nb, mycol, 0, npcol );
   for( k = 0; k < depth; k++ )
   {
      (void) HPL_pdupdate( NULL, NULL, panel[k], nn );
      (void) HPL_pdpanel_disp(  &panel[k] );
   }
   (void) HPL_pdpanel_disp( &panel[depth] );

   if( panel ) free( panel );
/*
 * End of HPL_pdgesvK2
 */
}
