MPI=-DMPI
MPICC= mpicxx
LDFLAGS = -lm

CC_FILES   := dcnn.cpp main.cpp matrix.cpp parse_file.cpp instrument.cpp
LOGS	   := logs

all: cnn-seq cnn-mpi

###########################################################

INSTRUMENT=1
DEBUG=0

OBJDIR=objs
CXX=g++ -m64
WFLGAS= -Wall -Wextra -Werror
CXXFLAGS=-O3 -g -std=c++1y -DTRACK=$(INSTRUMENT)

OBJS=$(OBJDIR)/main.o $(OBJDIR)/matrix.o $(OBJDIR)/dcnn.o $(OBJDIR)/parse_file.o $(OBJDIR)/instrument.o

.PHONY: dirs clean

default: all

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ cnn-seq cnn-mpi $(LOGS) *.ppm

cnn-seq: dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)


cnn-mpi: dirs $(OBJS)
		$(MPICC) $(CXXFLAGS) $(MPI) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

