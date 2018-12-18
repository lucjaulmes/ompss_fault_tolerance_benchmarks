CXX=g++
MCXX=smpcxx --cxx=$(CXX)
CXXFLAGS=-Wall -O3 -std=gnu++11

CC=gcc
MCC=smpcc --cc=$(CC)
CFLAGS=-Wall -O$(O) -std=gnu11 -I$(CATCHROI_HOME)/include -fno-optimize-sibling-calls
LDFLAGS=-L$(CATCHROI_HOME)/lib -Wl,-rpath,$(CATCHROI_HOME)/lib
LDLIBS+=-lm -lcatchroi
O=3

CATCHROI_HOME?=$(dir $(abspath $(dir $(lastword $(MAKEFILE_LIST)))))libcatchroi

OMPSS_FLAGS=--ompss --Wn,-Wno-unused-result,-Wno-unused-local-typedefs,-Wno-unused-but-set-variable,-Wno-discarded-array-qualifiers -Wno-unknown-pragmas -k -D_Float128=__float128

default: kmeans bmptopoints generate-random
perf:kmeans
debug:kmeans_debug
instr:kmeans_instr

%_instr:OMPSS_FLAGS+=--instrument
%_debug:OMPSS_FLAGS+=--debug
%_debug:CFLAGS+=-g

%:%.cc
	$(LINK.cc) $< $(LDLIBS) -o $@

kmeans: kmeans_perf
	@mv $< $@

kmeans_%: kmeans.c
	$(MCC) $(CFLAGS) $(OMPSS_FLAGS) $(CPPFLAGS) $(LDFLAGS) $(TARGET_ARCH) $< $(LDLIBS) -o $@

clean:
	rm -fv kmeans kmeans_debug kmeans_instr bmptopoints generate-random smpcxx_*.cc *~
