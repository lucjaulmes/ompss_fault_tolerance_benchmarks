TARGET:=stream
OMP_SRC:=stream.c

include ../common.mk

CFLAGS+=-ffast-math -fno-finite-math-only -fno-optimize-sibling-calls -Wno-sign-compare
CPPFLAGS+=-DVALIDATE=1 -DVERBOSE=1
LDLIBS+= -lm
