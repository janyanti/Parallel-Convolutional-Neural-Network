EXECUTABLE := cnn-omp
CU_FILES   :=
CU_DEPS    :=
CC_FILES   := dcnn.cpp main.cpp matrix.cpp  parse_file.cpp
LOGS	   := logs

all: $(EXECUTABLE)

###########################################################

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -g -std=c++1y -fopenmp -DOMP

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))


OBJS=$(OBJDIR)/main.o $(OBJDIR)/matrix.o $(OBJDIR)/dcnn.o $(OBJDIR)/parse_file.o

.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS) *.ppm

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDLIBS) $(LDFRAMEWORKS)


$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

