CC=gcc
CFLAGS=-g -O3 -fPIC -Wall -Wextra -std=gnu11
ARFLAGS=rcs

LIBS:=libcatchroi.a libcatchroi.so
HEADERS:=catchroi.h

make:$(LIBS)
%.so:LDFLAGS+=-shared -pthread

CATCHROI_HOME?=$(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DESTDIR?=$(CATCHROI_HOME)

ifeq ($(INJECT),1)
INJECT_OBJ=inject_err.o
else
INJECT_OBJ=
CPPFLAGS+=-DDISABLE_ERROR_INJECTION
endif

lib%.a:%.o $(INJECT_OBJ)
	$(AR) $(ARFLAGS) $@ $^

lib%.so:%.o $(INJECT_OBJ)
	$(LINK.o) $^ -o $@

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
