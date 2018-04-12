#include "DecisionTree.h"
#include <fstream>
#include <sstream>
#include <string>

vector<int> readX() {
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

vector<vector<double>> readY() {
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

int main(int argv, char **argc) {
  printf("Main. \n");
  vector<int> labels = readX();
  vector<vector<double>> features = readY();

  for (auto i = features.begin(); i != features.end(); ++i) {
    cout << "feature: " << endl;
    vector<double> cur = *i;
    for (auto j = cur.begin(); j != cur.end(); ++j) {
      cout << (*j) << " ";
    }
    cout << "\n";
  }
  return 0;
}
