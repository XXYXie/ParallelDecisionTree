// From https://github.com/DELTA37/ParallelDecisionTree
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
  // cout << "histNode->dataX[0][4]: " << histNode->dataX[0][0][4] << endl;

  q.push(histNode);
  // cout << "after push histNode" << endl;

  while (!q.empty()) {
    // cout << "q size: " << q.size() << endl;
    HistNode *curHistNode = q.front();
    q.pop();
    // cout << "after q.pop()" << endl;
    // cilk_for
    FixedHistogram** allHist[W];
    cilk_for (int i = 0; i < W; ++i) {
      // compress data.
      // cout << "worker i: " << i << endl;
      // cout << "data size: " << curHistNode->dataY[i].size() << endl;
      // for (auto iter = curHistNode->dataY[i].begin(); iter != curHistNode->dataY[i].end(); ++iter) {
      //   cout << "y: " << *iter << endl;
      // }
      // cout << "worker X: dataX[i][0][4]" << curHistNode->dataX[i][0][4] << endl;
      // for (int w = 0; w != curHistNode->dataX[i].size(); ++w) {
      //   for (int z = 0; z != curHistNode->dataX[i][w].size(); ++z) {
      //     cout << curHistNode->dataX[i][w][z] << endl;
      //   }
      // }
      // cout << "X end: " << endl;

      FixedHistogram **workerHist = compressData(
          curHistNode->dataX[i], curHistNode->dataY[i], numUniqueY, dimension);

      allHist[i] = workerHist;
      // cout << "workerHist begin: " << endl;
      // for (int j = 0; j < dimension; ++j) {
      //   for (int k = 0; k < numUniqueY; ++k) {
      //     // cout << "y: " << k << endl;
      //     for (auto iter = workerHist[j][k].data.begin(); iter != workerHist[j][k].data.end(); ++iter) {
      //       // cout << "first: " << iter->first << " second: " << iter->second << endl;
      //     }
      //   }
      // }
      // cout << "workerHist end: " << endl;
    }

    // cout << "after compress data" << endl;

    FixedHistogram **array = allHist[0];
    // master worker.
    cilk_for (int i = 0; i < dimension; ++i) {
      cilk_for (int j = 0; j < numUniqueY; ++j) {
        for (int k = 1; k < W; ++k) {
          // Merge k, i, j.
          // cout <<s i << " " << j << " " << k << endl;
          // cout << "before merge" << endl;
          array[i][j].merge(allHist[k][i][j]);
          // cout << "after merge" << endl;

        }
      }
    }


    // cout << "after merge array" << endl;
    // Entropy.
    int yCount[numUniqueY];
    double sum = 0.0;
    for (int j = 0; j < numUniqueY; ++j) {
      yCount[j] = array[0][j].sum(std::numeric_limits<double>::infinity());
      sum += yCount[j];
    }

    double entropy = 0.0;
    // cout << "sum: " << sum << endl;
    for (int j = 0; j < numUniqueY; ++j) {
      if (yCount[j] != 0) {
        double p = yCount[j] * 1.0 / sum;
        // cout << " yCount[j]: " << yCount[j] << endl;
        entropy += - p * log2(p);
      }
    }

    // cout << "entropy: " << entropy << endl;
    if (abs(entropy) < THRESHOLD) {
      // prediction is the index of the largest count.
      // cout << "thres" << endl;
      curHistNode->node->prediction = 1 + distance(yCount, max_element(yCount, yCount + numUniqueY));
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
        // cout << "after merge before uniform" << endl;
        // Uniform.
        // cout << "histArray[i]" << endl;
        // for (auto iter = histArray[i].data.begin(); iter != histArray[i].data.end(); ++iter) {
        //   cout << "key: " << iter->first << " , value: " << iter->second << endl;
        // }
        // cout << "end histArray[i]\n" << endl;
        vector<double> uniformResult = histArray[i].uniform(Bt);
        // cout << "after uniform" << endl;
        // cout << "uniform result begin " << endl;
        // for (auto k = uniformResult.begin(); k != uniformResult.end(); ++k) {
        //   cout << *k << endl;
        // }
        // cout << "uniform result end " << endl;

        int uniformsize = uniformResult.size();
        cilk_for (int ku = 0; ku < uniformsize; ++ku) {
          double k = uniformResult[ku];
        // for (auto k = uniformResult.begin(); k != uniformResult.end(); ++k) {
          // cout << "uniform result *k: " << *k << endl;
          double leftCount[numUniqueY];
          double rightCount[numUniqueY];
          double totalCount[numUniqueY];
          double sumLeftCount = 0.0;
          double sumRightCount = 0.0;
          double sumTotalCount = 0.0;
          // cout << "total count < left count ?????" << endl;
          for (int m = 0; m < numUniqueY; ++m) {
            // cout << "before sum:" << endl;
            leftCount[m] = array[i][m].sum(k);
            // cout << "*k: "<< *k << endl;
            // cout << "after sum" << endl;
            sumLeftCount += leftCount[m];
            totalCount[m] = array[i][m].sum(std::numeric_limits<double>::infinity());
            // cout << "leftCount[m]: " << leftCount[m] << endl;
            // cout << "totalCount[m]: " << totalCount[m] << endl;
            sumTotalCount += totalCount[m];
            rightCount[m] = totalCount[m] - leftCount[m];
            sumRightCount += rightCount[m];
          }
          // cout << "out numUniqueY" << endl;
          double leftEntropy = 0.0;
          double rightEntropy = 0.0;

          // cout << "start entropy for loop " << endl;
          // cout << "sumRightCount " << sumRightCount << endl;
          for (int m = 0; m < numUniqueY; ++m) {
            if (leftCount[m] != 0){
              double p = leftCount[m]/sumLeftCount;
              leftEntropy += - p * log2(p);
            }
            if (rightCount[m] != 0) {
              double p = rightCount[m]/sumRightCount;
              // cout << "rightCount[m]: " << rightCount[m] << endl;
              // cout << "p: " << p << endl;
              rightEntropy += - p * log2(p);
            }
          }
          // cout << "out numUniqueY2" << endl;
          // cout << "left entropy: " << leftEntropy << endl;
          // cout << "right entropy: " << rightEntropy << endl;
          // cout << "sumLeftCount: " << sumLeftCount << endl;
          // cout << "sumRightCount: " << sumRightCount << endl;
          // cout << "sumTotalCount: " << sumTotalCount << endl;

          double curEntropy = -sumLeftCount / sumTotalCount * leftEntropy -
                           sumRightCount / sumTotalCount * rightEntropy;
          // cout << "curEntropy: " << curEntropy << endl;
          // cout << "i: " << i << " *k: " << *k << endl;
          if (curEntropy > maxEntropy) {
            maxEntropy = curEntropy;
            bestSplitVal = k;
            bestFeature = i;
          }
        }
      }

      // cout << "bestSplitVal: " << bestSplitVal << endl;
      // cout << "bestFeature: " << bestFeature << endl;
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
          // cout << "tmpLeft size: " << tmpLeft.size() << endl;
          // cout << "tmpRight size: " << tmpRight.size() << endl;

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
        // cout << "leftHistNode->dataY[i] size: " << leftHistNode->dataY[i].size() << endl;
        // cout << "rightHistNode->dataY[i] size: " << rightHistNode->dataY[i].size() << endl;
      }

      // push two nodes onto the queue
      q.push(leftHistNode);
      q.push(rightHistNode);
      // cout << "after push 2" << endl;
    }

  }
  cout << "numNode: " << numNode << endl;
  return root;

}

FixedHistogram **compressData(vector<vector<double>> x,
                                              vector<int> y, int uniqueY,
                                              int dimension) {
  // cout << "compress data x[0][4]: " << x[0][4] << endl;
  // vector<vector<FixedHistogram>> ans;
  int numData = y.size();
  // set<int> uniqueY;
  // for (int i = 0; i < numData; i++) {
  //   uniqueY.insert(y[i]);
  // }
  // int numUniqueY = uniqueY.size();
  FixedHistogram **array = new FixedHistogram*[dimension];
  // Empty historgrams.
  vector<double> points;
  for (int i = 0; i < dimension; ++i) {
    array[i] = new FixedHistogram[uniqueY];
    for (int j = 0; j < uniqueY; ++j) {
      array[i][j] = FixedHistogram(Bt, points);
    }
  }

  // cout << "after assign FixedHistogram" << endl;

  for (int i = 0; i < numData; ++i) {
    // cout << "i: " << i << endl;
    for (int j = 0; j < dimension; ++j) {
      // cout << "i: " << i << " j: " << j << endl;
      // cout << "y[i]: " << y[i] << endl;
      // cout << "x[j][i]: " << x[j][i] << endl;
      // cout << "before update" << endl;
      // cout << "j: " << j << " i: " << i << " x[j][i]: " << x[j][i] << endl;
      array[j][y[i]-1].update(x[j][i], 1);
      // cout << "after update"  << endl;
    }
  }

  // cout << "after update histogram" << endl;

  return array;
}

void FixedHistogram::_reduce(void) {
  int size = data.size();
  // cout << "data size: " << size << endl;
  if (size == 0) return;
  std::list<double> diff;
  std::transform(data.begin(), std::prev(data.end()), std::next(data.begin()), std::back_inserter(diff),
    [](std::pair<double, int> const& x, std::pair<double, int> const& y) -> double {
      return y.first - x.first;
    }
  );

  // cout << "diff size: " << diff.size() << endl;

  if (size <= B) {
    //cout << "size <= B" << endl;
    return;
  }

  for (int i = 0; i < size - B; ++i) {
    // cout << "reduce for data: " << endl;
    // for (auto iterator = data.begin(); iterator != data.end(); iterator++) {
    //   cout << iterator->first << " "  << iterator->second << endl;
    // }
    // cout << "data size: " << data.size() << endl;
    // cout << "diff size: " << diff.size() << endl;
    auto list_it = std::min_element(diff.begin(), diff.end());
    auto list_nit = std::next(list_it);
    auto list_pit = std::prev(list_it);

    int ind = std::distance(diff.begin(), list_it);
    // cout << "ind: " << ind << endl;

    auto min_it = std::next(data.begin(), ind);
    auto min_nit = std::next(min_it);
    auto min_nnit = std::next(min_nit);
    auto min_pit = std::prev(min_it);

    //cout << "min_nit is end: " << (min_nit == data.end()) << endl;

    double point = (min_it->first * min_it->second + min_nit->first * min_nit->second) / (min_it->second + min_nit->second);
    int value = min_it->second + min_nit->second;

    //cout << "point: " << point << " value: " << value << endl;
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
    // cout << "erase min_it" << endl;
    data.erase(min_it);
    // cout << "erase min_nit" << endl;
    // cout << "min_nit->second: " << min_nit->second << endl;
    data.erase(min_nit);
    //cout << "erase min_nit" << endl;
    // cout << "after erase erase min_nit" << endl;

    diff.clear();
    std::transform(data.begin(), std::prev(data.end()), std::next(data.begin()), std::back_inserter(diff),
      [](std::pair<double, int> const& x, std::pair<double, int> const& y) -> double {
        return y.first - x.first;
      }
    );
    // cout << "after clear diff" << endl;
  }
}

FixedHistogram::FixedHistogram(int _B, std::vector<double> const& points, double _eps) : B(_B), eps(_eps) {
  min_point = std::numeric_limits<double>::max();
  max_point = std::numeric_limits<double>::min();
  // assert(_B == points.size());
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
  // cout << "p: " << p << endl;
  data[p] += c;
  // cout << "data[p]: " << data[p] << endl;
  min_point = std::min(p, min_point);
  max_point = std::max(p, max_point);
  // // cout << "before reduce all data" << endl;
  // for (auto iter = data.begin(); iter != data.end(); ++iter) {
  //   cout << "first: " << iter->first << " second: " << iter->second << endl;
  // }
  // cout << "before reduce" << endl;

  _reduce();
  // cout << "after reduce" << endl;
}

void FixedHistogram::merge(FixedHistogram const& h) {
  for (auto it = h.data.begin(); it != h.data.end(); ++it) {
    data[it->first] += it->second;
    min_point = std::min(it->first, min_point);
    max_point = std::max(it->first, max_point);
  }
  // cout << "before merge reduce" << endl;
  _reduce();
  // cout << "after merge reduce" << endl;
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
