// FixedHistogram class is from https://github.com/DELTA37/ParallelDecisionTree
#include <vector>
#include <functional>
#include <map>
#include <list>
#include <utility>
#include <numeric>
#include <cassert>
#include <iostream>
#include <queue>

#include "DecisionTree.h"

#define W 2
#define THRESHOLD 0.01
#define Bt 700

using namespace std;

class FixedHistogram {
public:
  map<double, int> data;

  FixedHistogram(){};

  FixedHistogram(int _B, vector<double> const& points, double eps=1e-5);

  double sum(double const& b);

  void update(double const& p, int c=1);

  void merge(FixedHistogram const& h);

  vector<double> uniform(int _B);

private:
  /*
   * The article definition B == data.size()
   */
  int B;
  double eps;
  // map<double, int> data;

  /*
   * left and right edges of histogram. We should compute it during execution
   */
  double min_point, max_point;

  void _reduce(void);

};

struct HistNode {
  Node *node;
  vector<vector<double>> dataX[W];
  vector<int> dataY[W];
};

Node *histTree(vector<vector<double>> x, vector<int> y);
FixedHistogram **compressData(vector<vector<double>> x, vector<int> y,
                              int uniqueY, int dimension);
vector<int> evalHistTree(Node *node, vector<vector<double>> &xTe);
rfOutput* randomForestHist(vector<vector<double>> x, vector<int> y, int k, int nt);
vector<int> evalForestHist(rfOutput* forestOutput, int nt, vector<vector<double>> &xTe);
