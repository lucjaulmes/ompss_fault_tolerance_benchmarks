# target recipes. NB most objects are seq
OMP_SRC:=algorithms.cc heat.cc
SEQ_SRC:=algorithms_inner.cc matrix.cc misc.cc
TARGET:=heat


include ../common.mk

# override to force C++98
CXXFLAGS = -O$(O) -std=gnu++98 -Wall -Wextra -rdynamic -g -fno-optimize-sibling-calls
