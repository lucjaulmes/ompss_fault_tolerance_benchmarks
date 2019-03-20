TARGET:=miniAMR
OMP_SRC:=block.c comm.c comm_alt.c driver.c init.c main.c
SEQ_SRC:=move.c profile.c refine.c target.c util.c task_functions.c

include ../common.mk

CPPFLAGS += -Wno-implicit-fallthrough
LDLIBS += -lm #-lrt
