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


tuple<const double, const Question> Calculations::find_best_split(const Data& rows, const MetaData& meta) {
  double bestGain = 0.0;  // keep track of the best information gain
  auto bestQuestion = Question();  //keep track of the feature / value that produced it
  //std::cout << "Whoop whoop INIT" << std::endl;
  //const auto &overall_counts = classCounts(rows);
  //const float current_uncertainty = gini(overall_counts, rows.size());
	size_t n_features = rows.back().size() - 1;  //number of columns
	
	#pragma omp parallel for num_threads(5)
	for (size_t column = 0; column < n_features; column++) {
		std::string colType = meta.columnTypes[column];
		
		double candGain = 0.0;
		//std::string candidateThresh = " ";
		//double candidateLoss = 0.0;
		//double candidateTrueSize = 0.0;
		//double candidateFalseSize = 0.0;
		//ClassCounter candidateTrueCounts = overall_counts;
		//ClassCounter candidateFalseCounts = overall_counts;
		//tuple<std::string, double> bestThreshAndLoss;
		//std::cout << "Whoop whoop0" << std::endl;
		if (colType.compare("categorical") == 0) {
			auto[candidateThresh, candidateGain] = determine_best_threshold_cat(rows, column);
			//if (candidateTrueSize == 0 || candidateFalseSize == 0)
			//	continue;
			//const auto &candidateGain = info_gain(candidateTrueCounts, candidateFalseCounts, candidateTrueSize, candidateFalseSize, current_uncertainty);
			std::cout << "Gain: " << candidateGain << std::endl;
			//candGain = candidateGain;
			#pragma omp critical
			{
				if (candidateGain >= bestGain) {
					const Question q(column, candidateThresh);
					bestGain = candidateGain;
					bestQuestion = q;
				}
			}
		} else {
			auto[candidateThresh, candidateGain] = determine_best_threshold_numeric(rows, column);
			//if (candidateTrueSize == 0 || candidateFalseSize == 0)
			//	continue;
			//const auto &candidateGain = info_gain(candidateTrueCounts, candidateFalseCounts, candidateTrueSize, candidateFalseSize, current_uncertainty);
			std::cout << "Gain: " << candidateGain << std::endl;
			//candGain = candidateGain;
			#pragma omp critical
			{
				if (candidateGain >= bestGain) {
					const Question q(column, candidateThresh);
					bestGain = candidateGain;
					bestQuestion = q;
				}
			}
		}

		
		//std::cout << "Whoop whoop3" << std::endl;
	}
   //std::cout << "Whoop whoop END" << std::endl;
  return forward_as_tuple(bestGain, bestQuestion);
}

const double Calculations::gini(const ClassCounter& counts, double N) {
  double impurity = 1.0;
  
  for (const auto& [decision, freq]: counts) {
		double prob_of_lbl = freq / N;
		impurity -= std::pow(prob_of_lbl, 2.0f);
  }

  return impurity;
}

float Calculations::info_gain(const ClassCounter &true_counts, const ClassCounter &false_counts, double &true_size, double &false_size, float current_uncertainty) {
	const float p = static_cast<float>(true_size) / (true_size + false_size);
	return current_uncertainty - p * gini(true_counts, true_size) - (1 - p) * gini(false_counts, false_size);
}

//struct result {std::string bestThresh; double bestLoss; double bestTrueSize; double bestFalseSize; ClassCounter &bestTrueCounts; ClassCounter &bestFalseCounts;};

std::tuple<std::string, double> Calculations::determine_best_threshold_numeric(const Data& data, int col) {
  double bestLoss = std::numeric_limits<float>::infinity();
  std::string bestThresh;
	double totalSize = data.size();
	double totalTrue = 0;
	
	ClassCounter totalClassCounts = classCounts(data);
	ClassCounter incrementalTrueClassCounts;
	ClassCounter incrementalFalseClassCounts;
	
	Data sortedData = sort_numeric_data(data, col);
	double current_uncertainty = gini(totalClassCounts, totalSize);
	for (const auto& [decision, freq]: totalClassCounts) {
		incrementalTrueClassCounts[decision] = 0;
		incrementalFalseClassCounts[decision] = freq;
  }
	
	//tracker variables 
	double bestTrueSize;
	double bestFalseSize;
	ClassCounter bestTrueCounts;
	ClassCounter bestFalseCounts;
	std::string currentFeatureValue = sortedData.front().at(0);
  for (std::vector<std::string> row : sortedData) {
        if (row.at(0) == currentFeatureValue) {
					// add to the class counter where relevant
          incrementalTrueClassCounts.at(row.back())++;
					incrementalFalseClassCounts.at(row.back())--;
					totalTrue += 1;
        } else {
					// first compare change in gini value - we don't compare IG, since this is constance to S
					// rather we want the minimal gini value for the split such that IG(S) - IG(S_new) is maximal
					
					const double trueGini = gini(incrementalTrueClassCounts, totalTrue);
					const double falseGini = gini(incrementalFalseClassCounts, (totalSize-totalTrue));
					const double currentGini = (trueGini * totalTrue + falseGini * (totalSize-totalTrue)) / totalSize;

					if (currentGini < bestLoss) {
							bestLoss = currentGini;
							bestThresh = row.at(0);
							bestTrueSize = totalTrue;
							bestFalseSize = (totalSize-totalTrue);
							bestTrueCounts = incrementalTrueClassCounts;
							bestFalseCounts = incrementalFalseClassCounts;
							if (IsAlmostEqual(bestLoss, 0.0))
									break;
					}
					
					// then add to the class counter where relevant
					incrementalTrueClassCounts.at(row.back())++;
					incrementalFalseClassCounts.at(row.back())--;	

					// update the current feature value being tracked against
					currentFeatureValue = row.at(0);
        }
  }
	//std::cout << "Whoop whoop6" << std::endl;	
  const float p = static_cast<float>(bestTrueSize) / (bestTrueSize + bestFalseSize);
  bestLoss = current_uncertainty - p * gini(bestTrueCounts, bestTrueSize) - (1 - p) * gini(bestFalseCounts, bestFalseSize);

  return forward_as_tuple(bestThresh, bestLoss);
}

std::tuple<std::string, double> Calculations::determine_best_threshold_cat(const Data& data, int col) {
  double bestLoss = std::numeric_limits<float>::infinity();
  std::string bestThresh;

  ClassCounter totalClassCounts = classCounts(data);
	double totalSize = data.size();
	double current_uncertainty = gini(totalClassCounts, totalSize);
	// instantiate here
	ClassCounter incrementalCategoryCounts;
	ClassCounterPerCategory incrementalTrueClassCountsPerCategory;
	ClassCounterPerCategory incrementalFalseClassCountsPerCategory;
	
	//tracker variables 
	double bestTrueSize;
	double bestFalseSize;
	ClassCounter bestTrueCounts;
	ClassCounter bestFalseCounts;
	
  for (std::vector<std::string> row : data) {
		std::string decision = row.at(col);
		std::string outcome = row.back();
    if (incrementalCategoryCounts.find(decision) != std::end(incrementalCategoryCounts)) {
			incrementalCategoryCounts.at(decision)++;
			incrementalTrueClassCountsPerCategory[decision][outcome]++;
			incrementalFalseClassCountsPerCategory[decision][outcome]--;
    } else {
			incrementalCategoryCounts[decision] += 1;
			ClassCounter incrementalTrueClassCounts;
			ClassCounter incrementalFalseClassCounts;
			for (const auto& [decision, freq]: totalClassCounts) {
				incrementalTrueClassCounts[decision] = 0;
				incrementalFalseClassCounts[decision] = freq;
			}
			incrementalTrueClassCounts[outcome] += 1;
			incrementalFalseClassCounts[outcome] -= 1;
			incrementalTrueClassCountsPerCategory[decision] = incrementalTrueClassCounts;
			incrementalFalseClassCountsPerCategory[decision] = incrementalFalseClassCounts;
    }    
			
  }
		
	// now we iterate over each class instance (parallelizable) and determine which class
	// holds the minimal gini update value for the information gain calculation later
	#pragma omp parallel for num_threads(5)
	for(ClassCounter::iterator datIt = std::begin(incrementalCategoryCounts); datIt != std::end(incrementalCategoryCounts); datIt++) {
	//for (const auto& [category, catSize]: incrementalCategoryCounts) {
		// first compare change in gini value - we don't compare IG, since this is constance to S
		// rather we want the minimal gini value for the split such that IG(S) - IG(S_new) is maximal
		
		ClassCounter incrementalTrueClassCounts = incrementalTrueClassCountsPerCategory[datIt->first];
		ClassCounter incrementalFalseClassCounts = incrementalFalseClassCountsPerCategory[datIt->first];
		double totalTrue = datIt->second;
		const double trueGini = gini(incrementalTrueClassCounts, totalTrue);
		const double falseGini = gini(incrementalFalseClassCounts, (totalSize-totalTrue));
		const double currentGini = (trueGini * totalTrue + falseGini * (totalSize-totalTrue)) / totalSize;
		#pragma omp critical
		if (currentGini < bestLoss) {
				bestLoss = currentGini;
				bestThresh = datIt->first;
				bestTrueSize = totalTrue;
				bestFalseSize = (totalSize-totalTrue);
				bestTrueCounts = incrementalTrueClassCounts;
				bestFalseCounts = incrementalFalseClassCounts;
				if (IsAlmostEqual(bestLoss, 0.0))
						break;
		}
	}
	//std::cout << "Whoop whoop5" << std::endl;
  const float p = static_cast<float>(bestTrueSize) / (bestTrueSize + bestFalseSize);
  bestLoss = current_uncertainty - p * gini(bestTrueCounts, bestTrueSize) - (1 - p) * gini(bestFalseCounts, bestFalseSize);

  return forward_as_tuple(bestThresh, bestLoss);
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

/**
 * Sort data in descending order based on provided column as index
 *
 * @param data  Data object containing data
 * @param col column to use as sort index
 */
const Data Calculations::sort_numeric_data(const Data &data, int col) {

    Data sortedData;
    for (VecS row : data) {
        const VecS newRow{row.at(col), row.back()};
        sortedData.emplace_back(std::move(newRow));
    }
    Data *temp = (Data *) &sortedData;
    sort(temp->begin(), temp->end(), comparator);
    return sortedData;
}

/**
 * Comparator assuming that the index of a vector (first element) is the sort index
 * Comparator assumes comparison of ordinal/numeric data points. Uses std::stoi to convert to integer values.
 *
 * @param row1: vector row to compare with row2
 * @param row2: vector row to compare with row1
 */
bool Calculations::comparator(VecS &row1, VecS &row2) {
    return std::stoi(row1.front()) > std::stoi(row2.front());
}

