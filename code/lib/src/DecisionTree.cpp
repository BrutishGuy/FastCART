/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission. 
 * Written by Pieter Robberechts, 2019
 */

#include "DecisionTree.hpp"
#include <future>
#include <tuple>

using std::make_shared;
using std::shared_ptr;
using std::string;
using boost::timer::cpu_timer;

DecisionTree::DecisionTree(const DataReader& dr) : root_(Node()), dr_(dr) {
  std::cout << "Start building tree." << std::endl; cpu_timer timer;
  root_ = buildTree(dr_.trainData(), dr_.metaData());
  std::cout << "Done. " << timer.format() << std::endl;
}


const Node DecisionTree::buildTree(const Data& rows, const MetaData& meta) {
    auto[gain, question] = Calculations::find_best_split(rows, meta);
    if (gain == 0.0) {
		ClassCounter classCounter = Calculations::classCounts(rows);
		Leaf leaf(classCounter);
		Node leafNode(leaf);
		return leafNode;
    }

    const auto[true_rows, false_rows] = Calculations::partition(rows, question);
    auto true_branch = std::async(std::launch::async, &DecisionTree::buildTree, this, std::cref(true_rows), std::cref(meta));
    auto false_branch = std::async(std::launch::async, &DecisionTree::buildTree, this, std::cref(false_rows), std::cref(meta));

    return Node(true_branch.get(), false_branch.get(), question);

}

void DecisionTree::print() const {
  print(make_shared<Node>(root_));
}

void DecisionTree::print(const shared_ptr<Node> root, string spacing) const {
  if (bool is_leaf = root->leaf() != nullptr; is_leaf) {
    const auto &leaf = root->leaf();
    std::cout << spacing + "Predict: "; Utils::print::print_map(leaf->predictions());
    return;
  }
  std::cout << spacing << root->question().toString(dr_.metaData().labels) << "\n";

  std::cout << spacing << "--> True: " << "\n";
  print(root->trueBranch(), spacing + "   ");

  std::cout << spacing << "--> False: " << "\n";
  print(root->falseBranch(), spacing + "   ");
}

void DecisionTree::test() const {
  TreeTest t(dr_.testData(), dr_.metaData(), root_);
}
