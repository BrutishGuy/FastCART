/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission. 
 * Written by Pieter Robberechts, 2019
 */

#ifndef DECISIONTREE_CALCULATIONS_HPP
#define DECISIONTREE_CALCULATIONS_HPP

#include <tuple>
#include <vector>
#include <string>
#include <unordered_map>
#include <boost/timer/timer.hpp>
#include "Question.hpp"
#include "Utils.hpp"

using ClassCounter = std::unordered_map<std::string, int>;
using ClassCounterPerCategory = std::unordered_map<std::string, ClassCounter>;  // map<featureValue, classCounter>

namespace Calculations {

std::tuple<const Data, const Data> partition(const Data &data, const Question &q);

const double gini(const ClassCounter& counts, double N);

float info_gain(const ClassCounter &true_counts, const ClassCounter &false_counts, double &true_size, double &false_size, float current_uncertainty);

std::tuple<const double, const Question> find_best_split(const Data &rows, const MetaData &meta);

tuple<std::string, double, double, double, ClassCounter, ClassCounter> determine_best_threshold_numeric(const Data &data, int col);

tuple<std::string, double, double, double, ClassCounter, ClassCounter> determine_best_threshold_cat(const Data &data, int col);

const ClassCounter classCounts(const Data &data);

const Data sort_numeric_data(const Data &data, int col);

bool comparator(VecS &row1, VecS &row2);

} // namespace Calculations

#endif //DECISIONTREE_CALCULATIONS_HPP
