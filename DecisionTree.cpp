#include "DecisionTree.h"
#include <set>
#include <algorithm>
#include <math.h>

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

using namespace std;

#define MAXDEPTH 4

EntropySplitOutput *entropySplit(double *xTr, int *yTr, vector<double> &weights,
                                 vector<int> &featureIndex, int numData) {
  int indexArr[featureIndex.size()];
  for (int i = 0; i < featureIndex.size(); ++i) {
    indexArr[i] = featureIndex[i];
  }

  set<int> uniqueY;
  for (int i = 0; i < numData; ++i) {
    uniqueY.insert(yTr[i]);
  }

  int bestFeature = *featureIndex.begin();
  double bestSplitVal = 0.0;
  double maxEntropy = - std::numeric_limits<double>::infinity();

  for (int iter = 0; iter < featureIndex.size(); ++iter) {
  // for (auto iter = featureIndex.begin(); iter != featureIndex.end(); ++iter) {
    // int i = *iter;
    int i = indexArr[iter];
    // cout << "i: " << i << endl;
    // TODO: sort and get indexes.
    double leftEntropy = 0.0;
    double rightEntropy = 0.0;

    for (int j = 0; j < numData - 1; ++j) {
      // cout << "i: " << i << " j: " << j << endl;
      // cout << "xTr[i*numData + j]: " << xTr[i*numData + j] << endl;
      if (xTr[i*numData + j + 1] != xTr[i*numData + j]) {
      // if (xTr[i][j + 1] != xTr[i][j]) {
        for (set<int>::iterator it = uniqueY.begin(); it != uniqueY.end(); ++it) {
          int curY = *it;
          int leftYCount = 0;
          int rightYCount = 0;
          for (int k = 0; k <= j; ++k) {
            if (yTr[k] == curY) leftYCount++;
          }
          for (int k = j + 1; k < numData; ++k) {
            if (yTr[k] == curY) rightYCount++;
          }
          double pLeft = 1.0 * leftYCount / (j+1);
          double pRight = 1.0 * rightYCount / (numData - j - 1);
          if (leftYCount == 0) {
            leftEntropy = 0;
          } else {
            leftEntropy += -pLeft * log2 (pLeft);
          }
          if (rightYCount == 0) {
            rightEntropy = 0;
          } else {
            rightEntropy += -pRight * log2 (pRight);
          }
          // cout << "left: " << leftEntropy << " right: " << rightEntropy << endl;
        }

        double curEntropy = - 1.0 * j / numData * leftEntropy - 1.0 * (numData - j) / numData * rightEntropy;
        // cout << "curEntropy: " << curEntropy << endl;
        if (curEntropy > maxEntropy) {
          maxEntropy = curEntropy;
          // bestSplitVal = (xTr[i][j + 1] + xTr[i][j]) / 2;
          bestSplitVal = (xTr[i*numData + j] + xTr[i*numData + j + 1]) / 2;
          bestFeature = i;
          // cout << "cur best feature: " << bestFeature << endl;
        }
      }
    }
  }
  EntropySplitOutput *output = new EntropySplitOutput();
  output->feature = bestFeature;
  output->splitVal = bestSplitVal;
  return output;
}

// Recursively build
Node *buildTree(Node *parent, int depth, const vector<int> &labels,
                const vector<vector<double>> &features,
                const vector<int> &index, vector<int> &featureIndex,
                vector<double> &weights) {
  // x, y values.
  int indexSize = index.size();
  int featureIndexSize = featureIndex.size();
  int featureSize = features.size();
  int labelSize = labels.size();
  if (depth >= MAXDEPTH || featureIndexSize == 0) {
    return NULL;
  }

  int *y = (int *)malloc(labelSize * sizeof(int));
  double *x = (double *)malloc(featureSize * labelSize * sizeof(double));

  set<int> setY;
  for (auto iter = index.begin(); iter != index.end(); ++iter) {
    int m = *iter;
    y[m] = labels[m];
    setY.insert(labels[m]);
  }

  if (setY.size() == 1) return NULL;

  for (auto iter = featureIndex.begin(); iter != featureIndex.end(); ++iter) {
    for (auto iter2 = index.begin(); iter2 != index.end(); ++iter2) {
      int j = *iter2;
      x[(*iter) * labels.size() + j] = features[*iter][j];
    }
  }

  // TODO: check x.

  EntropySplitOutput *splitOutput =
      entropySplit(x, y, weights, featureIndex, labelSize);
  // EntropySplitOutput *splitOutput = new EntropySplitOutput();
  delete []y;
  delete []x;

  int selectedFeatureIndex = splitOutput->feature;
  double splitVal = splitOutput->splitVal;

  Node *node = new Node();
  node->parent = parent;
  node->prediction = 0; // TODO: mode()
  node->featureIndex = selectedFeatureIndex;
  node->cutoff = splitVal;

  vector<int> leftIndex, rightIndex;
  vector<double> leftWeights, rightWeights;
  for (auto iter = index.begin(); iter != index.end(); ++iter){
    int i = *iter;
    if (features[selectedFeatureIndex][i] <= splitVal) {
      leftIndex.push_back(i);
      leftWeights.push_back(weights[i]);
    } else {
      rightIndex.push_back(i);
      rightWeights.push_back(weights[i]);
    }
  }

  const vector<int> leftIndexConst = leftIndex;
  const vector<int> rightIndexConst = rightIndex;

  std::vector<int>::iterator position = std::find(featureIndex.begin(), featureIndex.end(), selectedFeatureIndex);
  if (position != featureIndex.end())  featureIndex.erase(position);

  // cilk_spawn
  node->left = cilk_spawn buildTree(node, depth + 1, labels, features,
                                    leftIndexConst, featureIndex, leftWeights);
  node->right = buildTree(node, depth + 1, labels, features, rightIndexConst,
                          featureIndex, rightWeights);
  cilk_sync;
  return node;
}
