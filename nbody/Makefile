# Options
#
# BIGO=N2|NlogN|N
BIGO?=N2
# VERSION=deep_offload_global|deep_offload_partial|intel_offload|iomp|ompss|ompss_opencl
VERSION?=ompss
# MPI=0|1
MPI.=0


# override some stuff for MPI
ifeq ($(MPI),1)
CC:=mpicc
endif


# Main target
TARGET:=nbody_$(VERSION)_$(BIGO)


# Source files, based on version and compiler
vpath %.c src kernels
OMP_SRC:=main.c nbody_$(VERSION).c
SEQ_SRC:=common.c

ifeq ($(VERSION),ompss_opencl)
SEQ_SRC+=kernel_ocl.c
else
OMP_SRC+=kernel_smp.c
endif

ifeq ($(CC),icc)
OMP_SRC+=iomp_syms.c
endif


# core
include ../common.mk

CPPFLAGS+=-Iinclude -DBIGO=$(BIGO) -Dalign_value=aligned
LDLIBS+=-lm


# Extra libraries for MPI
ifeq ($(MPI),1)
#MPI_HOME ?= /usr/lib64/mpi/gcc/openmpi
MPI_HOME ?= /apps/INTEL/2017.4/impi/2017.3.196/intel64

CPPFLAGS += -I$(MPI_HOME)/include -DUSE_MPI=1
LDLIBS += -lmpi
LDFLAGS += -L$(MPI_HOME)/lib64 -L$(MPI_HOME)/lib
endif
