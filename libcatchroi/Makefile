CC=gcc
CFLAGS=-g -O3 -fPIC -Wall -Wextra -std=gnu11
ARFLAGS=rcs

LIBS:=libcatchroi.a libcatchroi.so libinject.a libinject.so
HEADERS:=catchroi.h

make:$(LIBS)
%.so:LDFLAGS+=-shared
libinject.so:LDLIBS+=-pthread -lm

CATCHROI_HOME?=$(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DESTDIR?=$(CATCHROI_HOME)

libcatchroi.so libcatchroi.a: EXTRA_OBJ:=
libinject.so libinject.a: EXTRA_OBJ:=interpose.c

lib%.a:%.o $(EXTRA_OBJ)
	$(AR) $(ARFLAGS) $@ $^

lib%.so:%.o $(EXTRA_OBJ)
	$(LINK.o) $^ $(LDLIBS) -o $@

clean:
	@rm -vf $(LIBS)

install:$(LIBS) $(HEADERS)
	@mkdir -p $(DESTDIR)/lib $(DESTDIR)/include
	@cp -v $(LIBS) $(DESTDIR)/lib
	@cp -v $(HEADERS) $(DESTDIR)/include

uninstall:
	@rm -v $(addprefix $(DESTDIR)/lib/, $(LIBS)) $(addprefix $(DESTDIR)/include/, $(HEADERS))
	@rmdir --ignore-fail-on-non-empty $(DESTDIR)/lib $(DESTDIR)/include

.PHONY: make clean install uninstall


XED=1
ifeq ($(XED)@$(shell uname -p),1@x86_64)
XED_PATH?=$(HOME)/.local

libinject%:CPPFLAGS+=-I$(XED_PATH)/include -DHAVE_XED
libinject%:LDFLAGS+=-L$(XED_PATH)/lib -Wl,-rpath,$(XED_PATH)/lib
libinject%:LDLIBS+=-lxed
endif
