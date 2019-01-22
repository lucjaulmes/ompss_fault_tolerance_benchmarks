TARGET:=blackscholes
OMP_SRC:=blackscholes.c

include ../common.mk

ERR_CHK?=0
CPPFLAGS += -DERR_CHK=$(ERR_CHK)
LDLIBS += -lm


# some more targets
extras: inputgen
