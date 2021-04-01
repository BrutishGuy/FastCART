/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission. 
 * Written by Pieter Robberechts, 2019
 */

#include <cmath>
#include <algorithm>
#include <iterator>
#include <omp.h>
#include "Calculations.hpp"
#include "Utils.hpp"

using std::tuple;
using std::pair;
using std::forward_as_tuple;
using std::vector;
using std::string;
using std::unordered_map;

tuple<const Data, const Data> Calculations::partition(const Data& data, const Question& q) {
  Data true_rows;
  Data false_rows;
  
  for (const auto &row: data) {
    if (q.solve(row))
      true_rows.push_back(row);
    else
      false_rows.push_back(row);
  }

  return forward_as_tuple(true_rows, false_rows);
}

float Calculations::info_gain(const Data &true_rows, const Data &false_rows, float current_uncertainty) {
	const double &true_size = true_rows.size();
	const double &false_size = false_rows.size();
    const float p = static_cast<float>(true_size) / (true_size + false_size);
    const auto &true_counts = classCounts(true_rows);
	const auto &false_counts = classCounts(false_rows);
	return current_uncertainty - p * gini(true_counts, true_size) - (1 - p) * gini(false_counts, false_size);
}


tuple<const double, const Question> Calculations::find_best_split(const Data& rows, const MetaData& meta) {
  double best_gain = 0.0;  // keep track of the best information gain
  auto best_question = Question();  //keep track of the feature / value that produced it
  
  const auto &overall_counts = classCounts(rows)
  const float &current_uncertainty = gini(overall_counts, rows.size());
    size_t n_features = rows.back().size() - 1;  //number of columns

    #pragma omp parallel for num_threads(5)
    for (size_t column = 0; column < n_features; column++) {
        const auto values = unique_values(rows, column);


        for (const auto &val: values) {
            const Question q(column, val);

            const auto& [true_rows, false_rows] = partition(rows, q);

            if (true_rows.empty() || false_rows.empty())
                continue;

            const auto &gain = info_gain(true_rows, false_rows, current_uncertainty);

            #pragma omp critical
            {
                if (gain >= best_gain) {
                    best_gain = gain;
                    best_question = q;
                }
            }
        }
    }
	
  return forward_as_tuple(best_gain, best_question);
}

const double Calculations::gini(const ClassCounter& counts, double N) {
  double impurity = 1.0;
  
  for (const auto& [decision, freq]: counts) {
        double prob_of_lbl = freq / N;
        impurity -= std::pow(prob_of_lbl, 2.0f);
    }

	
  return impurity;
}

tuple<std::string, double> Calculations::determine_best_threshold_numeric(const Data& data, int col) {
  double best_loss = std::numeric_limits<float>::infinity();
  std::string best_thresh;

  //TODO: find the best split value for a discrete ordinal feature
  return forward_as_tuple(best_thresh, best_loss);
}

tuple<std::string, double> Calculations::determine_best_threshold_cat(const Data& data, int col) {
  double best_loss = std::numeric_limits<float>::infinity();
  std::string best_thresh;

  //TODO: find the best split value for a categorical feature
  return forward_as_tuple(best_thresh, best_loss);
}

VecS Calculations::unique_values(const Data &data, size_t column) {
    VecS unique_vals;

    ClassCounter counter;
    for (const auto &rows: data) {
        const string &decision = rows.at(column);
        counter[decision] += 0;
    }

    unique_vals.reserve(counter.size());

    std::transform(begin(counter), std::end(counter), std::back_inserter(unique_vals),
                   [](const auto &kv) { return kv.first; });
    return unique_vals;
}

const ClassCounter Calculations::classCounts(const Data& data) {
  ClassCounter counter;
  for (const auto& rows: data) {
    const string decision = *std::rbegin(rows);
    if (counter.find(decision) != std::end(counter)) {
      counter.at(decision)++;
    } else {
      counter[decision] += 1;
    }
  }
  return counter;
}
