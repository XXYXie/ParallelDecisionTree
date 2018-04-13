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

// vector<int> labels;
// vector<vector<double>> features;

int main(int argv, char **argc) {
  printf("Main. \n");
  vector<int> labels = readY();
  vector<vector<double>> features = readX();

  vector<int> origIndex;
  for (int i = 0; i < labels.size(); ++i) {
    origIndex.push_back(i);
  }

  vector<int> origFeatureIndex;
  for (int j = 0; j < features.size(); ++j) {
    origFeatureIndex.push_back(j);
  }

  vector<double> weights;
  for (int i = 0; i < labels.size(); ++i) {
    weights.push_back(1.0);
  }

  buildTree(NULL, 0, labels, features, origIndex, origFeatureIndex, weights);

  return 0;
}
