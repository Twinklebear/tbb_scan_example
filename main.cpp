#include <vector>
#include <iostream>
#include "scan.h"

int main(int argc, char **argv) {
	if (argc == 1) {
		std::cout << "Usage: " << argv[0] << " (array of ints)\n";
		return 1;
	}
	std::vector<int> data;
	for (int i = 1; i < argc; ++i) {
		data.push_back(std::atoi(argv[i]));
	}

	std::cout << "Input:\n";
	for (const auto &d : data) {
		std::cout << d << ", ";
	}
	std::cout << "\n";

	std::vector<int> inclusive_result;
	int inclusive_sum = inclusive_scan(data, 0, inclusive_result, std::plus<int>());
	std::cout << "Inclusive Sum: " << inclusive_sum << "\n"
		<< "Inclusive Scan results:\n";
	for (const auto &r : inclusive_result) {
		std::cout << r << ", ";
	}
	std::cout << "\n";

	std::vector<int> exclusive_result;
	int exclusive_sum = exclusive_scan(data, 0, exclusive_result, std::plus<int>());
	std::cout << "Exclusive Sum: " << exclusive_sum << "\n"
		<< "Exclusive Scan results:\n";
	for (const auto &r : exclusive_result) {
		std::cout << r << ", ";
	}
	std::cout << "\n";

	return 0;
}

