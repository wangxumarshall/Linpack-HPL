#  
#  -- High Performance Computing Linpack Benchmark (HPL)                
#     SDC Detection Module Makefile                                    
#  
include Make.inc
#
# ######################################################################
#
INCdep           = \
   $(INCdir)/hpl_misc.h   $(INCdir)/hpl_blas.h   $(INCdir)/hpl_auxil.h \
   $(INCdir)/hpl_pmisc.h  $(INCdir)/hpl_grid.h   $(INCdir)/hpl_comm.h  \
   $(INCdir)/hpl_pauxil.h $(INCdir)/hpl_panel.h  $(INCdir)/hpl_pfact.h \
   $(INCdir)/hpl_pgesv.h  $(INCdir)/hpl_sdc.h
#
## Object files ########################################################
#
HPL_sdcobj       = \
   HPL_sdc_checksum.o     HPL_sdc_verify.o       HPL_sdc_report.o       \
   HPL_sdc_inject.o
#
## Targets #############################################################
#
all     : lib 
#
lib     : lib.grd
#
lib.grd : $(HPL_sdcobj)
	$(ARCHIVER) $(ARFLAGS) $(HPLlib) $(HPL_sdcobj)
	$(RANLIB) $(HPLlib)
	$(TOUCH) lib.grd
#
# ######################################################################
#
HPL_sdc_checksum.o       : ../HPL_sdc_checksum.c       $(INCdep)
	$(CC) -o $@ -c $(CCFLAGS)  ../HPL_sdc_checksum.c
HPL_sdc_verify.o         : ../HPL_sdc_verify.c         $(INCdep)
	$(CC) -o $@ -c $(CCFLAGS)  ../HPL_sdc_verify.c
HPL_sdc_report.o         : ../HPL_sdc_report.c         $(INCdep)
	$(CC) -o $@ -c $(CCFLAGS)  ../HPL_sdc_report.c
HPL_sdc_inject.o         : ../HPL_sdc_inject.c         $(INCdep)
	$(CC) -o $@ -c $(CCFLAGS)  ../HPL_sdc_inject.c
#
# ######################################################################
#
clean            :
	$(RM) *.o *.grd
#
# ######################################################################
