/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission. 
 * Written by Pieter Robberechts, 2019
 */

#include "DecisionTree.hpp"
#include "Utils.hpp"
#include <future>
//#include <boost/thread.hpp>
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

DecisionTree::DecisionTree(const DataReader &dr, const std::vector<size_t> &samples) : root_(Node()), dr_(dr) {
    std::cout << "Start building tree as part of bagging...." << std::endl;
    cpu_timer timer;

    Data sample_data;
    sample_data.reserve(samples.size());
    for (auto index : samples) {
        sample_data.emplace_back(std::move(dr_.trainData().at(index)));
    }
	
		root_ = buildTree(sample_data, dr_.metaData());
    std::cout << "Done with building tree as part of bagging.... " << timer.format() << std::endl;
}

const Node DecisionTree::buildTree(const Data& rows, const MetaData& meta) {
		//std::cout << "HUR DUR build INIT" << std::endl;
    auto[gain, question] = Calculations::find_best_split(rows, meta);
    if (IsAlmostEqual(gain, 0.0)) {
			ClassCounter classCounter = Calculations::classCounts(rows);
			Leaf leaf(classCounter);
			return Node(leaf);
    }
		//std::cout << "HUR DUR build 1" << std::endl;
    const auto[true_rows, false_rows] = Calculations::partition(rows, question);
		int depth = 0;
    auto true_branch = std::async(std::launch::async, &DecisionTree::buildTreeStandard, this, std::cref(true_rows), std::cref(meta), std::cref(depth));
    auto false_branch = std::async(std::launch::async, &DecisionTree::buildTreeStandard, this, std::cref(false_rows), std::cref(meta), std::cref(depth));
		
		return Node(true_branch.get(), false_branch.get(), question);
		
		//auto true_branch = buildTree(true_rows, meta);
    //auto false_branch = buildTree(false_rows, meta);
		/* Node *true_branch = new Node;
		Node *false_branch = new Node;
		
		if (rows.size() < 5000) {
			std::cout << "Going down" << std::endl;
			std::thread buildTrueTree([this, &true_rows, &meta, true_branch]() {
					*true_branch = buildTreeStandard(true_rows, meta);
			});
			std::thread buildFalseTree([this, &false_rows, &meta, false_branch]() {
					*false_branch = buildTreeStandard(false_rows, meta);
			});	
			buildTrueTree.join();
			buildFalseTree.join();
		} else {
			std::thread buildTrueTree([this, &true_rows, &meta, true_branch]() {
					*true_branch = buildTree(true_rows, meta);
			});
			std::thread buildFalseTree([this, &false_rows, &meta, false_branch]() {
					*false_branch = buildTree(false_rows, meta);
			});
			buildTrueTree.join();
			buildFalseTree.join();
		}



		return Node(*true_branch, *false_branch, question); */
		//delete true_branch;
		//delete false_branch;

}

const Node DecisionTree::buildTreeStandard(const Data& rows, const MetaData& meta, int depth) {
		//std::cout << "HUR DUR build INIT" << std::endl;
    auto[gain, question] = Calculations::find_best_split(rows, meta);
    if (IsAlmostEqual(gain, 0.0)) {
			ClassCounter classCounter = Calculations::classCounts(rows);
			Leaf leaf(classCounter);
			return Node(leaf);
    }
		std::cout << "HUR DUR build 2" << std::endl;
    const auto[true_rows, false_rows] = Calculations::partition(rows, question);
		depth += 1;
		auto true_branch = buildTreeStandard(true_rows, meta, depth);
    auto false_branch = buildTreeStandard(false_rows, meta, depth);
		return Node(std::move(true_branch), std::move(false_branch), question);

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
