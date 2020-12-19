EXECUTABLE := fast_shared

CU_FILES   := cufast_sharedmem.cu

CU_DEPS    :=

CC_FILES   := cpu_fast.cpp

all: $(EXECUTABLE) $(REFERENCE)

LOGS	   := logs

###########################################################

OBJDIR=objs
CXX=g++ -m64
# CXXFLAGS=-g -O3 -Wall `pkg-config --cflags --libs opencv`
CXXFLAGS=-O3 -Wall `pkg-config --cflags --libs opencv` -fopenmp
INCLUDEFLAGS=-l/usr/local/cuda-10.2/targets/x86_64-linux/include
OMPLDFLAGS=-lm
LDFLAGS=-L/usr/local/cuda-10.2/lib64/ -lcudart
# OMP=-fopenmp 
# LDFLAGS=-L/usr/local/cuda-8.0/lib64/ -lcudart
NVCC=nvcc
# NVCCFLAGS=-g -G -O3 -m64 --gpu-architecture compute_61
NVCCFLAGS= -O3 -m64 --gpu-architecture compute_61


OBJS=$(OBJDIR)/cpu_fast.o  $(OBJDIR)/cufast_sharedmem.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(LOGS)


$(EXECUTABLE): dirs $(OBJS)
		$(CXX)  -o $@ $(OBJS) $(CXXFLAGS) $(LDFLAGS) $(OMPLDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
