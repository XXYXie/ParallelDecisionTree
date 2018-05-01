#ifndef TREE_h
#define TREE_h

#include <iostream>
#include <vector>
using namespace std;

struct Node {
  // Prediction at this node.
  int prediction;
  // Index of feature to cut.
  int featureIndex;
  // Cutoff value.
  double cutoff;
  // Left and right child.
  Node *left;
  Node *right;
  // Parent of this node. Don't need this.
  // Node *parent;
};

struct EntropySplitOutput {
  // Best feature to split.
  int feature;
  // Value to split on.
  double splitVal;
  // Loss of best splitVal.
  // double loss;
};

struct rfOutput {
  Node** nodes;
  vector<int> selectedFeatures;
};

void buildTree(Node *parent, int depth, vector<int> &labels,
               vector<vector<double>> &features, vector<int> &index,
               vector<int> &featureIndex);
vector<int> evalTree(Node *node, vector<vector<double> > &xTe);
vector<int> sampleWithReplacement(int total, int size);
rfOutput* randomForest(vector<vector<double>> x, const vector<int> y, int k, int nt);
#endif
