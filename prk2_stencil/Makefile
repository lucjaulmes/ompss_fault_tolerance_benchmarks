OMP_SRC:=stencil.c
TARGET:=prk2_stencil

include ../common.mk

CFLAGS+=-Wno-array-bounds
CPPFLAGS+=-DVALIDATE=1 -DVERBOSE
LDLIBS+=-lm
