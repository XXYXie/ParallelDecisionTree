CC=gcc
CXX=g++

CFLAGS=-c -Wall -std=c++11

all: decisionTree

decisionTree: Main.o DecisionTree.o
	$(CXX) Main.o DecisionTree.o -o decisionTree

Main.o: Main.cpp
	$(CXX) $(CFLAGS) Main.cpp

DecisionTree.o: DecisionTree.cpp
	$(CXX) $(CFLAGS) DecisionTree.cpp
clean:
	-rm -f *.o
