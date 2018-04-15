#include "DecisionTree.h"
#include <set>
#include <algorithm>
#include <math.h>
#include <unordered_map>

// #include <cilk/cilk.h>
// #include <cilk/cilk_api.h>

using namespace std;

#define MAXDEPTH 6

vector<int> sortIndexes(vector<double> &v) {
  // initialize original index locations
  vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](int i1, int i2) {return v[i1] < v[i2];});

  return idx;
}

int mode(int *y, const vector<int> &index) {
  unordered_map<int, int> hashMap;
  for (auto iter = index.begin(); iter != index.end(); ++iter) {
    // cout << "iter: " << *iter << " ";
    cout << y[*iter] << " ";
    hashMap[y[*iter]] += 1;
  }

  int curMax = 0;
  int ansY = y[0];
  for (auto iter = hashMap.begin(); iter != hashMap.end(); ++iter) {
    if (iter->second > curMax) {
      curMax = iter->second;
      ansY = iter->first;
    }
  }
  cout << "\nmode: " << ansY << endl;
  return ansY;
}

EntropySplitOutput *entropySplit(vector<vector<double>> &xTr, vector<int> &yTr, vector<double> &weights,
                                 vector<int> &featureIndex, int numData) {
  numData = yTr.size();
  int indexArr[featureIndex.size()];
  for (int i = 0; i < featureIndex.size(); ++i) {
    indexArr[i] = featureIndex[i];
  }

  set<int> uniqueY;
  // cout << "before sort" << endl;
  for (int i = 0; i < numData; i++) {
    // cout << yTr[i] << " ";
    uniqueY.insert(yTr[i]);
  }
  // cout << "\nbefore sort end" << endl;
  int bestFeature = *featureIndex.begin();
  double bestSplitVal = 0.0;
  double maxEntropy = - std::numeric_limits<double>::infinity();

  for (int i = 0; i < featureIndex.size(); ++i) {
  // for (auto iter = featureIndex.begin(); iter != featureIndex.end(); ++iter) {
    // int i = *iter;
    // cout << "i: " << i << endl;

    vector<int> sortedIndex = sortIndexes(xTr[i]);

    cout << "feature i: " << i << endl;

    vector<double> sx;
    vector<int> sy;
    vector<double> sw;

    for (auto indexIter = sortedIndex.begin(); indexIter != sortedIndex.end(); ++indexIter) {
      int j = *indexIter;
      sx.push_back(xTr[i][j]);
      sy.push_back(yTr[j]);
      //sw.push_back(w[j]);
    }

    for (int j = 0; j < numData - 1; ++j) {
      // cout << "yaftersort: " << yTr[j] << endl;

      // cout << "j: " << j << endl;
      // cout << "xaftersort: " << xTr[i][j] << endl;
      // cout << j << " "<< yTr[j] << endl;
      if (sy[j + 1] != sy[j]) {
        double leftEntropy = 0.0;
        double rightEntropy = 0.0;
      // if (xTr[i][j + 1] != xTr[i][j]) {
        cout << "diff j: " << j << endl;
        cout << "sy[j]: " << sy[j] << " sy[j+1]: " << sy[j+1] << endl;
        for (set<int>::iterator it = uniqueY.begin(); it != uniqueY.end(); ++it) {
          int curY = *it;
          int leftYCount = 0;
          int rightYCount = 0;
          for (int k = 0; k <= j; ++k) {
            if (sy[k] == curY) leftYCount++;
          }
          for (int k = j + 1; k < numData; ++k) {
            if (sy[k] == curY) rightYCount++;
          }
          double pLeft = 1.0 * leftYCount / (j+1);
          double pRight = 1.0 * rightYCount / (numData - j - 1);
          cout << "pLeft:" << pLeft << " pRight: " << pRight << " ";
          if (leftYCount != 0) {
            leftEntropy += -pLeft * log2 (pLeft);
          }
          if (rightYCount != 0) {
            rightEntropy += -pRight * log2 (pRight);
          }
          cout << "left: " << leftEntropy << " right: " << rightEntropy << endl;
        }

        double curEntropy = - 1.0 * (j+1) / numData * leftEntropy - 1.0 * (numData - j - 1) / numData * rightEntropy;
        cout << "curEntropy: " << curEntropy << endl;
        if (curEntropy > maxEntropy) {
          maxEntropy = curEntropy;
          // bestSplitVal = (xTr[i][j + 1] + xTr[i][j]) / 2;
          bestSplitVal = (sx[j+1] + sx[j]) / 2;
          bestFeature = i;
          // cout << "cur best feature: " << bestFeature << endl;
        }
      }
    }
  }
  EntropySplitOutput *output = new EntropySplitOutput();
  output->feature = indexArr[bestFeature];
  output->splitVal = bestSplitVal;
  return output;
}

// Recursively build
void buildTree(Node *parent, int depth, const vector<int> &labels,
                const vector<vector<double>> &features,
                const vector<int> &index, vector<int> &featureIndex,
                vector<double> &weights) {
  // x, y values.
  int indexSize = index.size();
  int featureIndexSize = featureIndex.size();
  int featureSize = features.size();
  int labelSize = labels.size();

  int *y = (int *)malloc(labelSize * sizeof(int));
  double *x = (double *)malloc(featureSize * labelSize * sizeof(double));

  vector<int> yCopy;
  set<int> setY;
  for (auto iter = index.begin(); iter != index.end(); ++iter) {
    int m = *iter;
    y[m] = labels[m];
    setY.insert(labels[m]);
    yCopy.push_back(labels[m]);
  }

  if (depth > MAXDEPTH || setY.size() == 1) {
    parent->prediction = mode(y, index);
    return;
  }

  // if (parent == NULL) parent = new Node();

  vector<vector<double>> xCopy;
  for (auto iter = featureIndex.begin(); iter != featureIndex.end(); ++iter) {
    vector<double> curVector;
    for (auto iter2 = index.begin(); iter2 != index.end(); ++iter2) {
      int j = *iter2;
      x[(*iter) * labels.size() + j] = features[*iter][j];
      curVector.push_back(features[*iter][j]);
    }
    xCopy.push_back(curVector);
  }

  // TODO: check x.

  EntropySplitOutput *splitOutput =
      entropySplit(xCopy, yCopy, weights, featureIndex, labelSize);
  // EntropySplitOutput *splitOutput = new EntropySplitOutput();

  int selectedFeatureIndex = splitOutput->feature;
  cout << "selectedFeatureIndex: " << selectedFeatureIndex << endl;
  double splitVal = splitOutput->splitVal;
  cout << "numX: " << indexSize << endl;
  cout << "splitVal: " << splitVal << endl;
  // Node *node = new Node();
  // node->parent = parent;
  parent->prediction = mode(y, index);
  cout << "parent->prediction: " << parent->prediction << endl;
  parent->featureIndex = selectedFeatureIndex;
  parent->cutoff = splitVal;

  delete []y;
  delete []x;

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

  // std::vector<int>::iterator position = std::find(featureIndex.begin(), featureIndex.end(), selectedFeatureIndex);
  // if (position != featureIndex.end())  featureIndex.erase(position);

  // cilk_spawn
  parent->left = new Node();
  parent->right = new Node();
  buildTree(parent->left, depth + 1, labels, features,
                                    leftIndexConst, featureIndex, leftWeights);
  buildTree(parent->right, depth + 1, labels, features, rightIndexConst,
                          featureIndex, rightWeights);
  // cilk_sync;
  return;
}

vector<int> evalTree(Node *node, vector<vector<double>> &xTe) {
  vector<int> ans;
  int numX = xTe[0].size();
  for (int j = 0; j < numX; ++j) {
    Node *nodeCopy = node;
    while (nodeCopy->left != NULL && nodeCopy->right != NULL) {
      int curFeature = nodeCopy->featureIndex;
      if (xTe[curFeature][j] <= nodeCopy->cutoff) {
        nodeCopy = nodeCopy->left;
      } else {
        nodeCopy = nodeCopy->right;
      }
    }
    ans.push_back(nodeCopy->prediction);
  }
  return ans;
}
