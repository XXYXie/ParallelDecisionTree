# Parallel Decision Tree Building
### Sirui Li, Xueyan Xie
#### CSE 549 Final Project
We studied how to parallelize the Decision Tree Building procedure in a multi-processors setting. We first used Cilk spawn, sync, and parallel for to parallelize the algorithm. We tested its correctness as well as the time and speedup. Then we looked at an approximation algorithm, [Streaming Parallel Decision Tree](http://www.jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf), which is designed for classifying large data sets and streaming data. It assumes a setting where the size of data set is so large that it has to be stored distributively in different processors. We implemented SPDT in shared memory machine, measured its accuracy and performance and compared it with our recursive decision tree. At last, we applied the decision tree algorithm as a subroutine to build Bagging Forest. We saw some very interesting behaviors in terms of accurracy, time and speedup.

#### Recursive Decision Tree Performance
<img src="https://github.com/XXYXie/ParallelDecisionTree/blob/master/performance/recur_runtime.png" width="70%" height="70%">
<img src="https://github.com/XXYXie/ParallelDecisionTree/blob/master/performance/recur_speedup.png" width="70%" height="70%">

#### SPDT Performance
<img src="https://github.com/XXYXie/ParallelDecisionTree/blob/master/performance/hist_runtime.png" width="70%" height="70%">
<img src="https://github.com/XXYXie/ParallelDecisionTree/blob/master/performance/hist_speedup.png" width="70%" height="70%">
