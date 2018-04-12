#include "DecisionTree.h"
#include <fstream>
#include <sstream>
#include <string>

int main(int argv, char **argc) {
  printf("Main. \n");
  ifstream file("./data/xTr.csv");
  if (!file.is_open()) {
    cout << "Error opening file. " << endl;
  }

  string line;
  while (getline(file, line)) {
    stringstream lineStream(line);
    string cell;
    while (std::getline(lineStream, cell, ',')) {
      cout << cell << "\n";
    }
  }
  file.close();
  return 0;
}
