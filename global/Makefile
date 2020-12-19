EXECUTABLE := fast_global

CU_FILES   := cufast_globalmem.cu

CU_DEPS    :=

CC_FILES   := cpu_fast_global.cpp

all: $(EXECUTABLE) $(REFERENCE)

LOGS	   := logs

###########################################################

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS= -O3 -Wall `pkg-config --cflags --libs opencv`
INCLUDEFLAGS=-l/usr/local/cuda-10.2/targets/x86_64-linux/include
LDFLAGS=-L/usr/local/cuda-10.2/lib64/ -lcudart
# LDFLAGS=-L/usr/local/cuda-8.0/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61


OBJS=$(OBJDIR)/cpu_fast_global.o  $(OBJDIR)/cufast_globalmem.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(LOGS)


$(EXECUTABLE): dirs $(OBJS)
		$(CXX)  -o $@ $(OBJS) $(CXXFLAGS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
