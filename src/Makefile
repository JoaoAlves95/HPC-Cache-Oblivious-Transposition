SRCS = $(wildcard *.c)

PROGS = $(patsubst %.c,%,$(SRCS))

all: $(PROGS)

%: %.c
	gcc -O3 -o ../exe/$@ $<
clean: 
	rm -f exe/$(PROGS)
