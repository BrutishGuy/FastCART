/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission. 
 * Written by Pieter Robberechts, 2019
 */

#include "Bagging.hpp"
#include "DecisionTree.hpp"

using std::make_shared;
using std::shared_ptr;
using std::string;
using boost::timer::cpu_timer;

Bagging::Bagging(const DataReader& dr, const int ensembleSize, uint seed) : 
  dr_(dr), 
  ensembleSize_(ensembleSize),
  learners_({}) {
  random_number_generator.seed(seed);
  buildBag();
}


void Bagging::buildBag() {
  cpu_timer timer;
  std::vector<double> timings; 
  for (int i = 0; i < ensembleSize_; i++) {
		timer.start();
		std::uniform_int_distribution<int> uniform_sampler(0, dr_.trainData().size() - 1);
			std::vector<size_t> samples;
			samples.reserve(dr_.trainData().size());
			for (int i = 0; i < dr_.trainData().size(); i++) {
				samples.emplace_back(std::move(uniform_sampler(random_number_generator)));
			}
			DecisionTree dt = DecisionTree(dr_, samples);
    learners_.push_back(dt);
    auto nanoseconds = boost::chrono::nanoseconds(timer.elapsed().wall);
    auto seconds = boost::chrono::duration_cast<boost::chrono::seconds>(nanoseconds);
    timings.push_back(seconds.count());
  }
  float avg_timing = Utils::iterators::average(std::begin(timings), std::begin(timings) + std::min(5, ensembleSize_));
  std::cout << "Average timing: " << avg_timing << std::endl;
}

void Bagging::test() const {
  TreeTest t;
  float accuracy = 0;
  for (const auto& row: dr_.testData()) {
    static size_t last = row.size() - 1;
    std::vector<std::string> decisions;
    for (int i = 0; i < ensembleSize_; i++) {
      const std::shared_ptr<Node> root = std::make_shared<Node>(learners_.at(i).root_);
      const auto& classification = t.classify(row, root);
      decisions.push_back(Utils::tree::getMax(classification));
    }
    std::string prediction = Utils::iterators::mostCommon(decisions.begin(), decisions.end());
    if (prediction == row[last])
      accuracy += 1;
  }
  std::cout << "Total accuracy: " << (accuracy / dr_.testData().size()) << std::endl;
}


