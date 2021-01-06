SRCS = $(wildcard *.c)

PROGS = $(patsubst %.c,%,$(SRCS))

all: $(PROGS)

%: %.c
	mkdir ../exe/
	gcc -O3 -o ../exe/$@ $<
clean: 
	rm -f exe/$(PROGS)
