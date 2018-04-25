#include "DecisionTree.h"
#include "ktiming.h"
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

vector<int> readY(const string&  filename) {
  // Read in labels from y training data.
  vector<int> labels;
  ifstream file(filename.c_str());
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

vector<int> readYLetter(const string&  filename) {
  // Read in labels from y training data.
  vector<int> labels;
  ifstream file(filename.c_str());
  if (!file.is_open()) {
    cout << "Error opening file y. " << endl;
  }

  string line;
  while (getline(file, line)) {
    stringstream lineStream(line);
    string cell;
    while (std::getline(lineStream, cell, ',')) {
      const char *cur = cell.c_str();
      // cout << cur[0] << endl;
      labels.push_back(cur[0] - 65);
    }
  }
  file.close();
  return labels;
}

vector<vector<double> > readX(const string&  filename) {
  // Read in features from x training data.
  vector<vector<double> > features;
  ifstream filex(filename.c_str());
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

int main(int argv, char **argc) {
  printf("Main. \n");
  // const vector<int> labels = readY("./data/yTr.csv");
  // const vector<vector<double> > features = readX("./data/xTr.csv");
  const vector<int> labels = readYLetter("./data/letter_yTr100.csv");
  vector<vector<double> > features = readX("./data/letter_xTr100.csv");

  vector<int> origIndex = getOrigIndex(labels);

  vector<double> weights;
  for (int i = 0; i < labels.size(); ++i) {
    weights.push_back(1.0);
  }

  vector<int> origFeatureIndex;
  for (int j = 0; j < features.size(); ++j) {
    origFeatureIndex.push_back(j);
  }

  Node *root = new Node();

  clockmark_t begin_rm = ktiming_getmark();
  buildTree(root, 1, labels, features, origIndex, origFeatureIndex, weights);
  clockmark_t end_rm = ktiming_getmark();

  printf("Elapsed time in seconds: %f\n", ktiming_diff_sec(&begin_rm, &end_rm));

  // vector<int> labelsTest = readY("./data/yTe.csv");
  // vector<vector<double> > featuresTest = readX("./data/xTe.csv");
  // vector<int> labelsTest = readYLetter("./data/letter_yTe300.csv");
  // vector<vector<double> > featuresTest = readX("./data/letter_xTe300.csv");
  //
  // vector<int> pred = evalTree(root, featuresTest);
  // int numTest = labelsTest.size();
  // int same = 0;
  // for (int i = 0; i < numTest; ++i) {
  //   if (labelsTest[i] == pred[i]) {
  //     same++;
  //   }
  // }
  // double accuracy = 1.0 * same / numTest;
  // cout << "accuracy: " << accuracy << endl;
  return 0;
}
