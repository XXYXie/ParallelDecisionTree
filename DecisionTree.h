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
  // Parent of this node.
  Node *parent;
};

struct EntropySplitOutput {
  // Best feature to split.
  int feature;
  // Value to split on.
  double splitVal;
  // Loss of best splitVal.
  // double loss;
};
void buildTree(Node *parent, int depth, const vector<int> &labels,
                const vector<vector<double> > &features,
                const vector<int> &index, vector<int> &featureIndex,
                vector<double> &weights);
vector<int> evalTree(Node *node, vector<vector<double> > &xTe);
#endif
