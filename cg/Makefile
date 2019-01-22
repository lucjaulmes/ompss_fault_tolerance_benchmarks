TARGET:=cg
OMP_SRC:=cg.c main.c
SEQ_SRC:=$(filter-out $(OMP_SRC), $(wildcard *.c))

include ../common.mk

LDLIBS+=-lm
