// Parts of sum, merge, uniform and update are from https://github.com/DELTA37/ParallelDecisionTree
// We also fixed some of thier bugs.
#include "Histogram.h"
#include <cmath>
#include <algorithm>
#include <set>
#include <unordered_map>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

using namespace std;

int numNode = 1;
Node* histTree(vector<vector<double>> x, vector<int> y) {
  queue<HistNode*> q;
  Node *root = new Node();
  HistNode *histNode = new HistNode();
  histNode->node = root;

  int numData = y.size();
  int dimension = x.size();

  set<int> uniqueY;
  for (int i = 0; i < numData; i++) {
    uniqueY.insert(y[i]);
  }
  int numUniqueY = uniqueY.size();

// Split y to W workers
  cilk_for (int i = 0; i < W; ++i) {
    for (int j = i*numData/W; j < (i+1) * numData/W; ++j) {
      histNode->dataY[i].push_back(y[j]);
    }
  }

  // Split data to W workers.
  cilk_for (int i = 0; i < W; ++i) {
    for (int k = 0; k < dimension; ++k) {
      vector<double> tmp;
      for (int j = i*numData/W; j < (i+1) * numData/W; ++j) {
        tmp.push_back(x[k][j]);
      }
      histNode->dataX[i].push_back(tmp);
    }
  }

  q.push(histNode);

  while (!q.empty()) {
    HistNode *curHistNode = q.front();
    q.pop();
    FixedHistogram** allHist[W];
    cilk_for (int i = 0; i < W; ++i) {
      // compress data.

      FixedHistogram **workerHist = compressData(
          curHistNode->dataX[i], curHistNode->dataY[i], numUniqueY, dimension);

      allHist[i] = workerHist;
    }

    FixedHistogram **array = allHist[0];
    // master worker.
    cilk_for (int i = 0; i < dimension; ++i) {
      cilk_for (int j = 0; j < numUniqueY; ++j) {
        for (int k = 1; k < W; ++k) {
          // Merge k, i, j.
          array[i][j].merge(allHist[k][i][j]);
        }
      }
    }


    // Entropy.
    int yCount[numUniqueY];
    double sum = 0.0;
    for (int j = 0; j < numUniqueY; ++j) {
      yCount[j] = array[0][j].sum(std::numeric_limits<double>::infinity());
      sum += yCount[j];
    }

    double entropy = 0.0;
    for (int j = 0; j < numUniqueY; ++j) {
      if (yCount[j] != 0) {
        double p = yCount[j] * 1.0 / sum;
        entropy += - p * log2(p);
      }
    }

    if (abs(entropy) < THRESHOLD) {
      // prediction is the index of the largest count.
      curHistNode->node->prediction =
          1 + distance(yCount, max_element(yCount, yCount + numUniqueY));
    } else {
      numNode++;
      FixedHistogram histArray[dimension];
      int bestFeature = -1;
      double bestSplitVal = 0;
      double maxEntropy = - std::numeric_limits<double>::infinity();

      cilk_for (int i = 0; i < dimension; ++i) {
        histArray[i] = array[i][0];
        for (int j = 1; j < numUniqueY; ++j) {
          histArray[i].merge(array[i][j]);
        }

        vector<double> uniformResult = histArray[i].uniform(Bt);


        int uniformsize = uniformResult.size();
        cilk_for (int ku = 0; ku < uniformsize; ++ku) {
          double k = uniformResult[ku];
          double leftCount[numUniqueY];
          double rightCount[numUniqueY];
          double totalCount[numUniqueY];
          double sumLeftCount = 0.0;
          double sumRightCount = 0.0;
          double sumTotalCount = 0.0;

          for (int m = 0; m < numUniqueY; ++m) {
            leftCount[m] = array[i][m].sum(k);
            sumLeftCount += leftCount[m];
            totalCount[m] =
                array[i][m].sum(std::numeric_limits<double>::infinity());

            sumTotalCount += totalCount[m];
            rightCount[m] = totalCount[m] - leftCount[m];
            sumRightCount += rightCount[m];
          }
          double leftEntropy = 0.0;
          double rightEntropy = 0.0;

          for (int m = 0; m < numUniqueY; ++m) {
            if (leftCount[m] != 0){
              double p = leftCount[m]/sumLeftCount;
              leftEntropy += - p * log2(p);
            }
            if (rightCount[m] != 0) {
              double p = rightCount[m]/sumRightCount;
              rightEntropy += - p * log2(p);
            }
          }

          double curEntropy = -sumLeftCount / sumTotalCount * leftEntropy -
                           sumRightCount / sumTotalCount * rightEntropy;

          if (curEntropy > maxEntropy) {
            maxEntropy = curEntropy;
            bestSplitVal = k;
            bestFeature = i;
          }
        }
      }

      curHistNode->node->featureIndex = bestFeature;
      curHistNode->node->cutoff = bestSplitVal;

      Node *leftChild = new Node();
      Node *rightChild = new Node();
      curHistNode->node->left = leftChild;
      curHistNode->node->right = rightChild;

      HistNode *leftHistNode = new HistNode();
      leftHistNode->node = leftChild;
      HistNode *rightHistNode = new HistNode();
      rightHistNode->node = rightChild;

      cilk_for (int i = 0; i < W; ++i) { // for each worker
        // construct dataX
        for (int k = 0; k < dimension; ++k) {
          vector<double> tmpLeft;
          vector<double> tmpRight;
          for (int j = 0; j < curHistNode->dataY[i].size(); ++j) {
            if (curHistNode->dataX[i][bestFeature][j] <= bestSplitVal) {
              tmpLeft.push_back(curHistNode->dataX[i][k][j]);
            } else{
              tmpRight.push_back(curHistNode->dataX[i][k][j]);
            }
          }

          leftHistNode->dataX[i].push_back(tmpLeft);
          rightHistNode->dataX[i].push_back(tmpRight);
        }
        // construct dataY
        for (int j = 0; j < curHistNode->dataY[i].size(); ++j) {
          if (curHistNode->dataX[i][bestFeature][j] <= bestSplitVal) {
            leftHistNode->dataY[i].push_back(curHistNode->dataY[i][j]);
          } else {
            rightHistNode->dataY[i].push_back(curHistNode->dataY[i][j]);
          }
        }
      }

      // push two nodes onto the queue
      q.push(leftHistNode);
      q.push(rightHistNode);
    }

  }
  return root;

}

FixedHistogram **compressData(vector<vector<double>> x,
                                              vector<int> y, int uniqueY,
                                              int dimension) {
  int numData = y.size();

  FixedHistogram **array = new FixedHistogram*[dimension];
  // Empty historgrams.
  vector<double> points;
  for (int i = 0; i < dimension; ++i) {
    array[i] = new FixedHistogram[uniqueY];
    for (int j = 0; j < uniqueY; ++j) {
      array[i][j] = FixedHistogram(Bt, points);
    }
  }


  for (int i = 0; i < numData; ++i) {
    for (int j = 0; j < dimension; ++j) {
      array[j][y[i]-1].update(x[j][i], 1);
    }
  }

  return array;
}

void FixedHistogram::_reduce(void) {
  int size = data.size();
  if (size == 0) return;
  std::list<double> diff;
  std::transform(data.begin(), std::prev(data.end()), std::next(data.begin()), std::back_inserter(diff),
    [](std::pair<double, int> const& x, std::pair<double, int> const& y) -> double {
      return y.first - x.first;
    }
  );


  if (size <= B) {
    return;
  }

  for (int i = 0; i < size - B; ++i) {
    auto list_it = std::min_element(diff.begin(), diff.end());
    auto list_nit = std::next(list_it);
    auto list_pit = std::prev(list_it);

    int ind = std::distance(diff.begin(), list_it);

    auto min_it = std::next(data.begin(), ind);
    auto min_nit = std::next(min_it);
    auto min_nnit = std::next(min_nit);
    auto min_pit = std::prev(min_it);


    double point = (min_it->first * min_it->second + min_nit->first * min_nit->second) / (min_it->second + min_nit->second);
    int value = min_it->second + min_nit->second;

    // if (min_nnit != data.end()) {
    //   diff.insert(list_it, min_nnit->first - point);
    // }
    //
    // if (min_it != data.begin()) { // min_pit is a valid element in data
    //   diff.insert(list_it, point - min_pit->first);
    // }
    //
    // // if (list_it != diff.begin()) {
    // if (ind != 0) {
    //   cout << "erase list_pit" << endl;
    //   diff.erase(list_pit);
    // }
    // cout << "erase list_it" << endl;
    // diff.erase(list_it);
    // //cout << "after erase list_it" << endl;
    // if (list_nit != diff.end()) {
    //   cout << "erase list_nit" << endl;
    //   diff.erase(list_nit);
    // }

    data[point] = value;
    data.erase(min_it);
    data.erase(min_nit);

    diff.clear();
    std::transform(data.begin(), std::prev(data.end()), std::next(data.begin()), std::back_inserter(diff),
      [](std::pair<double, int> const& x, std::pair<double, int> const& y) -> double {
        return y.first - x.first;
      }
    );
  }
}

FixedHistogram::FixedHistogram(int _B, std::vector<double> const& points, double _eps) : B(_B), eps(_eps) {
  min_point = std::numeric_limits<double>::max();
  max_point = std::numeric_limits<double>::min();
  for (auto const& p : points) {
    data[p] += 1;
    min_point = std::min(p, min_point);
    max_point = std::max(p, max_point);
  }
}

double FixedHistogram::sum(double const& b) {
  if (b >= max_point) {
    double s = 0;
    for (auto iter=data.begin(); iter != data.end(); ++iter) {
      s += iter->second;
    }
    return s;
  } else if (b < min_point) {
    return 0;
  }

  auto nit = data.upper_bound(b);

  auto it = (nit != data.begin()) ? std::prev(nit) : data.end();
  double s = 0;
  double p_i, p_ni;
  int m_i, m_ni;
  std::tie(p_i, m_i) = (it != data.end()) ? *it : std::make_tuple(std::min(b, min_point - eps), 0);
  std::tie(p_ni, m_ni) = (nit != data.end()) ? *nit : std::make_tuple(std::max(max_point + eps, b), 0);

  double m_b = m_i + (m_ni - m_i) * (b - p_i) / (p_ni - p_i);
  s += (m_i + m_b) * (b - p_i) / (p_ni - p_i) / 2.0;

  for (auto _it = data.begin(); _it != it; ++_it) {
    s += _it->second;
  }
  s += m_i / 2.0;
  return s;
}

std::vector<double> FixedHistogram::uniform(int _B) {
  std::vector<double> s;
  std::vector<double> u;
  u.reserve(_B);
  s.reserve(B);
  // m is actually a FixedHistogram.sum([-inf, p_i])
  double sumed = 0;
  // cout << "data: " << endl;
  for (auto it = data.begin(); it != data.end(); ++it) {
    s.emplace_back(sumed + (it->second / 2.0));
    // cout << "s: " << sumed + (it->second / 2.0) << endl;
    sumed += it->second;
  }
  // cout << "min_point: " << min_point << endl;
  u.emplace_back(min_point);
  // cout << "max_point: " << max_point << endl;

  for (int j = 1; j < _B - 1; ++j) {
    // cout << "j: " << j << endl;
    double  b        = (sumed * j) / _B;
    auto    it       = std::prev(std::upper_bound(s.begin(), s.end(), b));
    auto    data_it  = std::next(data.begin(), std::distance(s.begin(), it));
    auto    data_nit = std::next(data_it);

    // cout << "data_it is end: " << (data_it == data.end()) << endl;
    // cout << "data_nit is end: " << (data_nit == data.end()) << endl;

    auto [p_i, m_i]   = *data_it;
    auto [p_ni, m_ni] = (data_nit != data.end()) ? *data_nit : std::tuple{max_point, 0};

    double s_i = *it;
    double d   = b - s_i;
    u.emplace_back(
      (std::abs(m_ni - m_i) < eps) ? p_i : (p_i + (p_ni - p_i) * (-m_i + std::sqrt(m_i * m_i + 2.0 * (m_ni - m_i) * d)) / (m_ni - m_i))
    );

    // double val = (std::abs(m_ni - m_i) < eps) ? p_i : (p_i + (p_ni - p_i) * (-m_i + std::sqrt(m_i * m_i + 2.0 * (m_ni - m_i) * d)) / (m_ni - m_i));
    // cout << "val: " << val << endl;
    // cout << "m_i: " << m_i << endl;
    // cout << "m_ni: " << m_ni << endl;
  }
  u.emplace_back(max_point);
  return u;
}

void FixedHistogram::update(double const& p, int c) {
  data[p] += c;
  min_point = std::min(p, min_point);
  max_point = std::max(p, max_point);

  _reduce();
}

void FixedHistogram::merge(FixedHistogram const& h) {
  for (auto it = h.data.begin(); it != h.data.end(); ++it) {
    data[it->first] += it->second;
    min_point = std::min(it->first, min_point);
    max_point = std::max(it->first, max_point);
  }
  _reduce();
}

vector<int> evalHistTree(Node *node, vector<vector<double>> &xTe) {
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

vector<int> sampleWithReplacementHist(int total, int size) {
  vector<int> ans;
  for (int i = 0; i < size; ++i) {
    int randomNum = rand() % total;
    ans.push_back(randomNum);
  }
  return ans;
}

rfOutput* randomForestHist(vector<vector<double>> x, vector<int> y, int k, int nt) {
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

  cilk_for (int i = 0; i < nt; ++i) {
    // Sample k feature without replacement.
    // vector<int> sampleFeature;
    // // obtain a time-based seed:
    // unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    // shuffle(featureIndex.begin(), featureIndex.end(), default_random_engine(seed));
    // for (int j = 0; j < k; ++j) {
    //   sampleFeature.push_back(featureIndex[j]);
    // }

    // Sample numData data.
    const vector<int> sampleData = sampleWithReplacementHist(numData, numData);
    vector<vector<double>> newX;
    vector<int> newY;
    for (int j = 0; j < numFeature; ++j) {
      vector<double> tmpX;
      for (auto n = sampleData.begin(); n != sampleData.end(); ++n) {
        tmpX.push_back(x[j][*n]);
      }
      newX.push_back(tmpX);
    }
    for (auto n = sampleData.begin(); n != sampleData.end(); ++n) {
      newY.push_back(y[*n]);
    }

    Node *node = histTree(newX, newY);
    ans[i] = node;
  }
  rfOutput *output = new rfOutput();
  output->nodes = ans;
  output->selectedFeatures = featureIndex;
  return output;
}

int modeHist(vector<int> &y, int size) {
  unordered_map<int, int> hashMap;
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
  return ansY;
}

vector<int> evalForestHist(rfOutput* forestOutput, int nt, vector<vector<double>> &xTe) {
  vector<vector<int>> ans;
  Node **treeArr = forestOutput->nodes;
  vector<int> selectedFeatures = forestOutput->selectedFeatures;

  for (int i = 0; i < nt; ++i) {
    Node *tree = treeArr[i];
    vector<int> treeEval = evalHistTree(tree, xTe);
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
    pred.push_back(modeHist(curColumn, numRow));
  }
  return pred;
}
