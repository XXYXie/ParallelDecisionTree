#include "DecisionTree.h"
#include <fstream>
#include <sstream>
#include <string>


vector<int> readY() {
  // Read in labels from y training data.
  vector<int> labels;
  ifstream file("./data/yTr.csv");
  if (!file.is_open()) {
    cout << "Error opening file y. " << endl;
  }

  string line;
  while (getline(file, line)) {
    stringstream lineStream(line);
    string cell;
    while (std::getline(lineStream, cell, ',')) {
      labels.push_back(stoi(cell));
    }
  }
  file.close();
  return labels;
}

vector<vector<double>> readX() {
  // Read in features from x training data.
  vector<vector<double>> features;
  ifstream filex("./data/xTr.csv");
  if (!filex.is_open()) {
    cout << "Error opening file x. " << endl;
  }

  string line;
  while (getline(filex, line)) {
    stringstream lineStream(line);
    string cell;
    vector<double> curFeature;
    while (std::getline(lineStream, cell, ',')) {
      curFeature.push_back(stod(cell));
    }
    features.push_back(curFeature);
  }
  filex.close();
  return features;
}

vector<int> getOrigIndex(const vector<int> labels ) {
  vector<int> origIndex;
  for (int i = 0; i < labels.size(); ++i) {
    origIndex.push_back(i);
  }
  return origIndex;
}
// vector<int> labels;
// vector<vector<double>> features;

int main(int argv, char **argc) {
  printf("Main. \n");
  const vector<int> labels = readY();
  const vector<vector<double>> features = readX();

  const vector<int> origIndex = getOrigIndex(labels);

  vector<int> origFeatureIndex;
  for (int j = 0; j < features.size(); ++j) {
    origFeatureIndex.push_back(j);
  }
  // cout << "orig feature index start: " << endl;
  // for (auto i = origFeatureIndex.begin(); i != origFeatureIndex.end(); ++i) {
  //   cout << *i << " ";
  // }
  // cout << "\norig feature index end " << endl;

  vector<double> weights;
  for (int i = 0; i < labels.size(); ++i) {
    weights.push_back(1.0);
  }

  buildTree(NULL, 1, labels, features, origIndex, origFeatureIndex, weights);

  return 0;
}
