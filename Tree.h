#ifndef TREE_h
#define TREE_h

#include <iostream>
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
