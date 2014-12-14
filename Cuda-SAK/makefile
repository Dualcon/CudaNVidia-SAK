# Easily adaptable makefile for Advanced Programming
# Note: remove comments (#) to activate some features
#
# author Vitor Carreira
# date 2010-09-26
# Updated: Patricio R. Domingues (added more gcc's warning related flags)
# date: 2014-11-04


# Libraries to include (if any)
LIBS=#-pthread 

# Compiler flags
CFLAGS=-Wall -Wextra -Wmissing-prototypes -Wformat -Wsequence-point  -Wswitch-default -Wshadow -Wunused-function -Wunused-parameter -Wunused-value  -Wunused-variable -Wcast-qual -Wconversion -Wstrict-prototypes -Wuninitialized -std=c99 -D_SVID_SOURCE

# Indentation flags
IFLAGS=-br -brs -npsl -ce -cli4


# Name of the executable
PROGRAM=cuda-sak

# Prefix for the gengetopt file (if gengetopt is used)
PROGRAM_OPT=cuda-sakgengetopt

# Object files required to build the executable
PROGRAM_OBJS=main.o debug.o memory.o functions.o ${PROGRAM_OPT}.o

# Clean and all are not files
.PHONY: clean all docs indent debugon

all: ${PROGRAM}

# compilar com depuracao
debugon: CFLAGS += -D SHOW_DEBUG -g
debugon: ${PROGRAM}

${PROGRAM}: ${PROGRAM_OBJS}
	${CC} -o $@ ${PROGRAM_OBJS} ${LIBS}

# Dependencies
main.o: main.c debug.h memory.h functions.o ${PROGRAM_OPT}.h
${PROGRAM_OPT}.o: ${PROGRAM_OPT}.c ${PROGRAM_OPT}.h

debug.o: debug.c debug.h
memory.o: memory.c memory.h
#cmdline.o: cmdline.c cmdline.h
functions.o: functions.c functions.h

#how to create an object file (.o) from C file (.c)
.c.o:
	${CC} ${CFLAGS} -c $<

# Generates command line arguments code from gengetopt configuration file
${PROGRAM_OPT}.h: ${PROGRAM_OPT}.ggo
	gengetopt < ${PROGRAM_OPT}.ggo --file-name=${PROGRAM_OPT}

clean:
	rm -f *.o core.* *~ ${PROGRAM} *.bak ${PROGRAM_OPT}.h ${PROGRAM_OPT}.c

docs: Doxyfile
	doxygen Doxyfile

Doxyfile:
	doxygen -g Doxyfile

indent:
	indent ${IFLAGS} *.c *.h
