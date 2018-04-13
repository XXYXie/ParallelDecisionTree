#include "DecisionTree.h"
#include <set>
#include <algorithm>
#define MAXDEPTH 4
using namespace std;

// Calculate entropy.
double entropy() { return -1; }

EntropySplitOutput *entropySplit(double **xTr , int *yTr , vector<double> weights) {
  EntropySplitOutput *output = new EntropySplitOutput();
  return output;
}

// Recursively build
Node *buildTree(Node *parent, int depth, vector<int> labels,
                vector<vector<double>> features, vector<int> index,
                vector<int> featureIndex, vector<double> weights) {
  // x, y values.
  int indexSize = index.size();
  int featureIndexSize = featureIndex.size();
  int y[indexSize];
  double x[featureIndexSize][indexSize];

  for (int i = 0; i < featureIndexSize; ++i) {
    for (int j = 0; j != indexSize; ++j) {
      x[i][j] = features[i][j];
    }
  }

  set<int> setY;
  for (int i = 0; i < featureIndexSize; ++i) {
    y[i] = labels[i];
    setY.insert(labels[i]);
  }

  // set<int> setY{std::begin(y), std::end(y)};

  if (depth >= MAXDEPTH || setY.size() == 1 || featureIndex.size() == 0) {
    return NULL;
  }

  // TODO: check x.

  for (int i = 0; i < featureIndexSize; ++i) {
    for (int j = 0; j != indexSize; ++j) {
      x[i][j] = features[i][j];
    }
  }

  for (int i = 0; i < featureIndexSize; ++i) {
    y[i] = labels[i];
  }

  EntropySplitOutput *splitOutput = entropySplit((double**)x, (int*)y, weights);

  int selectedFeatureIndex = splitOutput->feature;
  double splitVal = splitOutput->splitVal;

  Node *node = new Node();
  node->parent = parent;
  node->prediction = 0; // TODO: mode()
  node->featureIndex = selectedFeatureIndex;
  node->cutoff = splitVal;

  vector<int> leftIndex, rightIndex;
  vector<double> leftWeights, rightWeights;
  for (int i = 0; i < indexSize; ++i) {
    if (features[selectedFeatureIndex][i] <= splitVal) {
      leftIndex.push_back(index[i]);
      leftWeights.push_back(weights[i]);
    } else {
      rightIndex.push_back(index[i]);
      rightWeights.push_back(weights[i]);
    }
  }

  remove(featureIndex.begin(), featureIndex.end(), selectedFeatureIndex);
  node->left = buildTree(node, depth + 1, labels, features, leftIndex,
                         featureIndex, leftWeights);
  node->right = buildTree(node, depth + 1, labels, features, rightIndex,
                          featureIndex, rightWeights);
  return node;
}
