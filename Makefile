CC=/project/cec/class/cse539/tapir/build/bin/clang
CXX=/project/cec/class/cse539/tapir/build/bin/clang++
# CXX=g++
CILK_LIBS=/project/linuxlab/gcc/6.4/lib64

CFLAGS = -ggdb -O3 -fcilkplus
CXXFLAGS = -c -std=c++17 -O3 -fcilkplus
LIBS = -L$(CILK_LIBS) -Wl,-rpath -Wl,$(CILK_LIBS) -lcilkrts -lpthread -lrt -lm

INST_RTS_LIBS=/project/cec/class/cse539/inst-cilkplus-rts/lib
# INST_LIBS = -L$(INST_RTS_LIBS) -Wl,-rpath -Wl,$(INST_RTS_LIBS) -lcilkrts -lpthread -lrt -lm -ldl

all: decisionTree

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

# When compile histogram, use clang++. When compile DecisionTree, use g++
decisionTree: Main.o Histogram.o ktiming.o
# decisionTree: Main.o DecisionTree.o ktiming.o Histogram
#	$(CXX) Main.o DecisionTree.o -o decisionTree
	$(CXX) -o $@ $^ $(LIBS)

# Main.o: Main.cpp
# 	$(CXX) $(CFLAGS) Main.cpp
#
# DecisionTree.o: DecisionTree.cpp
# 	$(CXX) $(CFLAGS) DecisionTree.cpp
clean:
	-rm -f *.o
