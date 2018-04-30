#!/bin/sh

#$ -cwd
#$ -j yes
#$ -o result_his_15000_16.out

CILK_NWORKERS=16 ./decisionTree
CILK_NWORKERS=16 ./decisionTree
CILK_NWORKERS=16 ./decisionTree
CILK_NWORKERS=16 ./decisionTree
CILK_NWORKERS=16 ./decisionTree
