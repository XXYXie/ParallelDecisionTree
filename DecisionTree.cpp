#include "DecisionTree.h"
#include "sort.h"
#include <set>
#include <algorithm>
#include <math.h>
#include <unordered_map>
#include <chrono>
#include <random>
#include <atomic>

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

using namespace std;

#define MAXDEPTH 20

bool pairCompare(const pair<double, int> &firstElem,
                 const pair<double, int> &secondElem) {
  return firstElem.first < secondElem.first;
}

int mode(vector<int> &y, int size) {
  unordered_map<int, int> hashMap;
  // for (auto iter = index.begin(); iter != index.end(); ++iter) {
  //   // cout << y[*iter] << " ";
  //   hashMap[y[*iter]] += 1;
  // }
  for (int i = 0; i < size; ++i) {
    hashMap[y[i]] += 1;
  }

  int curMax = 0;
  int ansY = y[0];
  for (auto iter = hashMap.begin(); iter != hashMap.end(); ++iter) {
    if (iter->second > curMax) {
      curMax = iter->second;
      ansY = iter->first;
    }
  }
  // cout << "\nmode: " << ansY << endl;
  return ansY;
}

EntropySplitOutput *entropySplit(vector<vector<double>> &xTr, vector<int> &yTr,
                                 vector<int> &featureIndex) {
  int numData = yTr.size();

  set<int> uniqueY;
  for (int i = 0; i < numData; i++) {
    uniqueY.insert(yTr[i]);
  }

  atomic<int> bestFeature(*featureIndex.begin());
  atomic<double> bestSplitVal(0.0);
  atomic<double> maxEntropy (- std::numeric_limits<double>::infinity());
  // int bestFeature = *featureIndex.begin();
  // double bestSplitVal = 0.0;
  // double maxEntropy = - std::numeric_limits<double>::infinity();

  cilk_for (int featureI = 0; featureI < featureIndex.size(); ++featureI) {
    // int i = *iter;
    int i = featureIndex[featureI];
    // vector<int> sortedIndex = sortIndexes(xTr[i]);

    // a pair of x and its corresponding label y.
    vector<pair<double, int>> xyPair;
    for (int k = 0; k < numData; ++k) {
      double key = xTr[i][k];
      int val = yTr[k];
      xyPair.push_back(make_pair(key, val));
    }

    // From https://software.intel.com/en-us/articles/parallel-sorts-for-cilk-plus
    cilkpub::cilk_sort_in_place (xyPair.begin(), xyPair.end(), pairCompare);
    // cout << "numData: " << numData << endl;
    // for (auto iter = xyPair.begin(); iter != xyPair.end(); ++iter) {
    //   cout << "x: " << iter->first << " y: " << iter->second << endl;
    // }
    // vector<double> sx;
    // vector<int> sy;
    // vector<double> sw;

    // for (auto indexIter = sortedIndex.begin(); indexIter != sortedIndex.end(); ++indexIter) {
    //   int j = *indexIter;
    //   sx.push_back(xTr[i][j]);
    //   sy.push_back(yTr[j]);
    //   //sw.push_back(w[j]);
    // }

    cilk_for (int j = 0; j < numData - 1; ++j) {

      if (xyPair[j + 1].second != xyPair[j].second) {
        // cout << "j: " << j << endl;
        double leftEntropy = 0.0;
        double rightEntropy = 0.0;

        for (set<int>::iterator it = uniqueY.begin(); it != uniqueY.end(); ++it) {
          int curY = *it;
          int leftYCount = 0;
          int rightYCount = 0;
          for (int k = 0; k <= j; ++k) {
            if (xyPair[k].second == curY) leftYCount++;
          }
          for (int k = j + 1; k < numData; ++k) {
            if (xyPair[k].second == curY) rightYCount++;
          }
          double pLeft = 1.0 * leftYCount / (j+1);
          double pRight = 1.0 * rightYCount / (numData - j - 1);
          // if (leftYCount != 0 || rightYCount != 0) {
          //   cout << "leftCount:" << leftYCount << " rightCount: " << rightYCount << " ";
          // }
          // if (pLeft != 0 || pRight != 0) cout << "pLeft:" << pLeft << " pRight: " << pRight << " ";
          if (leftYCount != 0) {
            leftEntropy += -pLeft * log2 (pLeft);
          }
          if (rightYCount != 0) {
            rightEntropy += -pRight * log2 (pRight);
          }
          // cout << "entropy left: " << leftEntropy << " right: " << rightEntropy << endl;
        }

        double curEntropy = - 1.0 * (j+1) / numData * leftEntropy - 1.0 * (numData - j - 1) / numData * rightEntropy;
        // cout << "curEntropy: " << curEntropy << endl;
        if (curEntropy > maxEntropy) {
          maxEntropy = curEntropy;
          bestSplitVal = (xyPair[j+1].first + xyPair[j].first) / 2;
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
void buildTree(Node *currentNode, int depth, vector<int> &labels,
               vector<vector<double>> &features, vector<int> &index,
               vector<int> &featureIndex) {
  // x, y values.
  int indexSize = index.size();
  int featureSize = features.size();
  int labelSize = labels.size();

  // int *y = (int *)malloc(labelSize * sizeof(int));
  // double *x = (double *)malloc(featureSize * labelSize * sizeof(double));

  vector<int> yCopy;
  set<int> uniqueY;
  if (index.size() == 0) {
    currentNode->prediction = 30;
    return;
  }
  for (auto iter = index.begin(); iter != index.end(); ++iter) {
    int m = *iter;
    // y[m] = labels[m];
    uniqueY.insert(labels[m]);
    yCopy.push_back(labels[m]);
  }

  // y labels are the same. Return the uniqueY as the prediction of the current node
  if (uniqueY.size() == 1) {
    currentNode->prediction = *uniqueY.begin();
    return;
  }

  if (depth > MAXDEPTH) {
    currentNode->prediction = mode(yCopy, indexSize);
    return;
  }

  // if (currentNode == NULL) currentNode = new Node();

  vector<vector<double>> xCopy;
  for (auto iter = featureIndex.begin(); iter != featureIndex.end(); ++iter) {
    int i = *iter;
  // for (int i = 0; i < featureSize; ++i) {
    vector<double> curVector;
    for (auto iter2 = index.begin(); iter2 != index.end(); ++iter2) {
      int j = *iter2;
      // x[i * labels.size() + j] = features[i][j];
      curVector.push_back(features[i][j]);
    }
    xCopy.push_back(curVector);
  }

  EntropySplitOutput *splitOutput = entropySplit(xCopy, yCopy, featureIndex);

  int selectedFeatureIndex = splitOutput->feature;
  // cout << "selectedFeatureIndex: " << selectedFeatureIndex << endl;
  double splitVal = splitOutput->splitVal;
  // cout << "numX: " << indexSize << endl;
  // cout << "splitVal: " << splitVal << endl;
  // Node *node = new Node();
  // node->currentNode = currentNode;
  currentNode->prediction = mode(yCopy, indexSize);
  // cout << "currentNode->prediction: " << currentNode->prediction << endl;
  currentNode->featureIndex = selectedFeatureIndex;
  currentNode->cutoff = splitVal;

  // delete []y;
  // delete []x;
  vector<int> leftIndex, rightIndex;
  // vector<double> leftWeights, rightWeights;
  for (auto iter = index.begin(); iter != index.end(); ++iter){
    int i = *iter;
    if (features[selectedFeatureIndex][i] <= splitVal) {
      leftIndex.push_back(i);
      // leftWeights.push_back(weights[i]);
    } else {
      rightIndex.push_back(i);
      // rightWeights.push_back(weights[i]);
    }
  }
  // cout << "after left right index" << endl;
  // std::vector<int>::iterator position = std::find(featureIndex.begin(), featureIndex.end(), selectedFeatureIndex);
  // if (position != featureIndex.end())  featureIndex.erase(position);

  currentNode->left = new Node();
  currentNode->right = new Node();
  // cilk_spawn
  cilk_spawn buildTree(currentNode->left, depth + 1, labels, features,
                       leftIndex, featureIndex);
  buildTree(currentNode->right, depth + 1, labels, features, rightIndex,
            featureIndex);
  cilk_sync;
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

// Can't use std::sample because it requires gcc 7.1
vector<int> sampleWithReplacement(int total, int size) {
  vector<int> ans;
  for (int i = 0; i < size; ++i) {
    int randomNum = rand() % total;
    ans.push_back(randomNum);
  }
  return ans;
}

rfOutput* randomForest(const vector<vector<double>> x, const vector<int> y, int k, int nt) {
  Node** ans = (Node **)malloc(sizeof(Node*) * nt);
  vector<int> featureIndex;
  vector<int> index;
  int numFeature = x.size();
  int numData = y.size();

  for (int i = 0; i < numData; ++i) {
    index.push_back(i);
  }

  for (int i = 0; i < numFeature; ++i) {
    featureIndex.push_back(i);
  }

  cout << "before for loop" << endl;
  for (int i = 0; i < nt; ++nt) {
    // Sample k feature without replacement.
    // vector<int> sampleFeature;
    // // obtain a time-based seed:
    // unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    // shuffle(featureIndex.begin(), featureIndex.end(), default_random_engine(seed));
    // for (int j = 0; j < k; ++j) {
    //   sampleFeature.push_back(featureIndex[j]);
    // }

    // Sample numData data.
    const vector<int> sampleData = sampleWithReplacement(numData, numData);
    vector<vector<double>> newX;
    vector<int> newY;
    for (int j = 0; j < numFeature; ++j) {
      vector<double> tmpX;
      for (auto n = sampleData.begin(); n != sampleData.end(); ++n) {
        tmpX.push_back(x[j][*n]);
        cout << x[j][*n] << endl;
      }
      newX.push_back(tmpX);
    }
    cout << "after push new X" << endl;
    for (auto n = sampleData.begin(); n != sampleData.end(); ++n) {
      cout << y[*n] << endl;
      newY.push_back(y[*n]);
    }
    cout << "after push newY" << endl;

    Node *node = NULL;
    buildTree(node, 1, newY, newX, index, featureIndex);
    ans[i] = node;
    cout << "after build tree" << endl;
  }
  rfOutput *output = new rfOutput();
  output->nodes = ans;
  output->selectedFeatures = featureIndex;
  return output;
}

vector<int> evalForest(rfOutput* forestOutput, int nt, vector<vector<double>> &xTe) {
  vector<vector<int>> ans;
  Node **treeArr = forestOutput->nodes;
  vector<int> selectedFeatures = forestOutput->selectedFeatures;

  for (int i = 0; i < nt; ++i) {
    Node *tree = treeArr[i];
    vector<int> treeEval = evalTree(tree, xTe);
    ans.push_back(treeEval);
  }

  int numCol = ans[0].size();
  int numRow = ans.size();
  vector<int> pred;
  for (int j = 0; j < numCol; ++j) {
    vector<int> curColumn;
    for (int i = 0; i < numRow; ++i) {
      curColumn.push_back(ans[i][j]);
    }
    pred.push_back(mode(curColumn, numRow));
  }
  return pred;
}
