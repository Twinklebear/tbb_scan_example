#include <vector>
#include <iostream>
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>

template<typename T, typename Op>
T inclusive_scan(const std::vector<T> &in, const T &id, std::vector<T> &out, Op op) {
	out.resize(in.size(), 0);
	using range_type = tbb::blocked_range<size_t>;
	T sum = tbb::parallel_scan(range_type(0, in.size()), id,
		[&](const range_type &r, T sum, bool is_final_scan) {
			T tmp = sum;
			for (size_t i = r.begin(); i < r.end(); ++i) {
				tmp = op(tmp, in[i]);
				if (is_final_scan) {
					out[i] = tmp;
				}
			}
			return tmp;
		},
		[&](const T &a, const T &b) {
			return op(a, b);
		});
	return sum;
}

template<typename T, typename Op>
T exclusive_scan(const std::vector<T> &in, const T &id, std::vector<T> &out, Op op) {
	// Exclusive scan is the same as inclusive, but shifted by one
	out.resize(in.size() + 1, 0);
	using range_type = tbb::blocked_range<size_t>;
	T sum = tbb::parallel_scan(range_type(0, in.size()), id,
		[&](const range_type &r, T sum, bool is_final_scan) {
			T tmp = sum;
			for (size_t i = r.begin(); i < r.end(); ++i) {
				tmp = op(tmp, in[i]);
				if (is_final_scan) {
					out[i + 1] = tmp;
				}
			}
			return tmp;
		},
		[&](const T &a, const T &b) {
			return op(a, b);
		});
	out.pop_back();
	return sum;
}

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

