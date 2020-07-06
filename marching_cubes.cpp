#include <array>
#include <random>
#include <chrono>
#include <fstream>
#include <cmath>
#include <iostream>
#include <tbb/parallel_for.h>
#include "scan.h"

using namespace std::chrono;

// Edge and triangle tables for the cases of marching cubes
// From http://paulbourke.net/geometry/polygonise/ and
// https://graphics.stanford.edu/~mdfisher/MarchingCubes.html
const std::array<std::array<int, 16>, 256> tri_table = {
	std::array<int, 16>{-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  8,  3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 1,  9,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 8,  1,  9,  8,  3,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 2, 10,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  8,  3,  1,  2, 10, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 9,  2, 10,  9,  0,  2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 3,  2, 10,  3, 10,  8,  8, 10,  9, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 2,  3, 11, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{11,  0,  8, 11,  2,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 1,  9,  0,  2,  3, 11, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 2,  1,  9,  2,  9, 11, 11,  9,  8, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 3, 10,  1,  3, 11, 10, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 1,  0,  8,  1,  8, 10, 10,  8, 11, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  3, 11,  0, 11,  9,  9, 11, 10, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{11, 10,  9, 11,  9,  8, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 4,  7,  8, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 4,  3,  0,  4,  7,  3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 4,  7,  8,  9,  0,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 9,  4,  7,  9,  7,  1,  1,  7,  3, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 4,  7,  8,  1,  2, 10, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 4,  3,  0,  4,  7,  3,  2, 10,  1, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 2,  9,  0,  2, 10,  9,  4,  7,  8, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 3,  2,  7,  7,  9,  4,  7,  2,  9,  9,  2, 10, -1,  0,  0,  0},
	std::array<int, 16>{ 8,  4,  7,  3, 11,  2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 7, 11,  2,  7,  2,  4,  4,  2,  0, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 2,  3, 11,  1,  9,  0,  8,  4,  7, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 2,  1,  9,  2,  9,  4,  2,  4, 11, 11,  4,  7, -1,  0,  0,  0},
	std::array<int, 16>{10,  3, 11, 10,  1,  3,  8,  4,  7, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 4,  7,  0,  0, 10,  1,  7, 10,  0,  7, 11, 10, -1,  0,  0,  0},
	std::array<int, 16>{ 8,  4,  7,  0,  3, 11,  0, 11,  9,  9, 11, 10, -1,  0,  0,  0},
	std::array<int, 16>{ 7,  9,  4,  7, 11,  9,  9, 11, 10, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 4,  9,  5, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 8,  3,  0,  4,  9,  5, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  5,  4,  0,  1,  5, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 4,  8,  3,  4,  3,  5,  5,  3,  1, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 1,  2, 10,  9,  5,  4, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 4,  9,  5,  8,  3,  0,  1,  2, 10, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{10,  5,  4, 10,  4,  2,  2,  4,  0, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 4,  8,  3,  4,  3,  2,  4,  2,  5,  5,  2, 10, -1,  0,  0,  0},
	std::array<int, 16>{ 2,  3, 11,  5,  4,  9, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{11,  0,  8, 11,  2,  0,  9,  5,  4, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 5,  0,  1,  5,  4,  0,  3, 11,  2, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{11,  2,  8,  8,  5,  4,  2,  5,  8,  2,  1,  5, -1,  0,  0,  0},
	std::array<int, 16>{ 3, 10,  1,  3, 11, 10,  5,  4,  9, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 9,  5,  4,  1,  0,  8,  1,  8, 10, 10,  8, 11, -1,  0,  0,  0},
	std::array<int, 16>{10,  5, 11, 11,  0,  3, 11,  5,  0,  0,  5,  4, -1,  0,  0,  0},
	std::array<int, 16>{ 4, 10,  5,  4,  8, 10, 10,  8, 11, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 7,  9,  5,  7,  8,  9, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  9,  5,  0,  5,  3,  3,  5,  7, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 8,  0,  1,  8,  1,  7,  7,  1,  5, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 3,  1,  5,  3,  5,  7, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 7,  9,  5,  7,  8,  9,  1,  2, 10, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 1,  2, 10,  0,  9,  5,  0,  5,  3,  3,  5,  7, -1,  0,  0,  0},
	std::array<int, 16>{ 7,  8,  5,  5,  2, 10,  8,  2,  5,  8,  0,  2, -1,  0,  0,  0},
	std::array<int, 16>{10,  3,  2, 10,  5,  3,  3,  5,  7, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 9,  7,  8,  9,  5,  7, 11,  2,  3, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  9,  2,  2,  7, 11,  2,  9,  7,  7,  9,  5, -1,  0,  0,  0},
	std::array<int, 16>{ 3, 11,  2,  8,  0,  1,  8,  1,  7,  7,  1,  5, -1,  0,  0,  0},
	std::array<int, 16>{ 2,  7, 11,  2,  1,  7,  7,  1,  5, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{11,  1,  3, 11, 10,  1,  7,  8,  9,  7,  9,  5, -1,  0,  0,  0},
	std::array<int, 16>{11, 10,  1, 11,  1,  7,  7,  1,  0,  7,  0,  9,  7,  9,  5, -1},
	std::array<int, 16>{ 5,  7,  8,  5,  8, 10, 10,  8,  0, 10,  0,  3, 10,  3, 11, -1},
	std::array<int, 16>{11, 10,  5, 11,  5,  7, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{10,  6,  5, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  8,  3, 10,  6,  5, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 9,  0,  1,  5, 10,  6, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 8,  1,  9,  8,  3,  1, 10,  6,  5, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 6,  1,  2,  6,  5,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 6,  1,  2,  6,  5,  1,  0,  8,  3, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 5,  9,  0,  5,  0,  6,  6,  0,  2, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 6,  5,  2,  2,  8,  3,  5,  8,  2,  5,  9,  8, -1,  0,  0,  0},
	std::array<int, 16>{ 2,  3, 11, 10,  6,  5, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0, 11,  2,  0,  8, 11,  6,  5, 10, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  1,  9,  3, 11,  2, 10,  6,  5, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{10,  6,  5,  2,  1,  9,  2,  9, 11, 11,  9,  8, -1,  0,  0,  0},
	std::array<int, 16>{11,  6,  5, 11,  5,  3,  3,  5,  1, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{11,  6,  8,  8,  1,  0,  8,  6,  1,  1,  6,  5, -1,  0,  0,  0},
	std::array<int, 16>{ 0,  3, 11,  0, 11,  6,  0,  6,  9,  9,  6,  5, -1,  0,  0,  0},
	std::array<int, 16>{ 5, 11,  6,  5,  9, 11, 11,  9,  8, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 7,  8,  4,  6,  5, 10, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 3,  4,  7,  3,  0,  4,  5, 10,  6, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 6,  5, 10,  7,  8,  4,  9,  0,  1, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 5, 10,  6,  9,  4,  7,  9,  7,  1,  1,  7,  3, -1,  0,  0,  0},
	std::array<int, 16>{ 1,  6,  5,  1,  2,  6,  7,  8,  4, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 7,  0,  4,  7,  3,  0,  6,  5,  1,  6,  1,  2, -1,  0,  0,  0},
	std::array<int, 16>{ 4,  7,  8,  5,  9,  0,  5,  0,  6,  6,  0,  2, -1,  0,  0,  0},
	std::array<int, 16>{ 2,  6,  5,  2,  5,  3,  3,  5,  9,  3,  9,  4,  3,  4,  7, -1},
	std::array<int, 16>{ 4,  7,  8,  5, 10,  6, 11,  2,  3, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 6,  5, 10,  7, 11,  2,  7,  2,  4,  4,  2,  0, -1,  0,  0,  0},
	std::array<int, 16>{ 4,  7,  8,  9,  0,  1,  6,  5, 10,  3, 11,  2, -1,  0,  0,  0},
	std::array<int, 16>{ 6,  5, 10, 11,  4,  7, 11,  2,  4,  4,  2,  9,  9,  2,  1, -1},
	std::array<int, 16>{ 7,  8,  4, 11,  6,  5, 11,  5,  3,  3,  5,  1, -1,  0,  0,  0},
	std::array<int, 16>{ 0,  4,  7,  0,  7,  1,  1,  7, 11,  1, 11,  6,  1,  6,  5, -1},
	std::array<int, 16>{ 4,  7,  8,  9,  6,  5,  9,  0,  6,  6,  0, 11, 11,  0,  3, -1},
	std::array<int, 16>{ 7, 11,  4, 11,  9,  4, 11,  5,  9, 11,  6,  5, -1,  0,  0,  0},
	std::array<int, 16>{10,  4,  9, 10,  6,  4, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{10,  4,  9, 10,  6,  4,  8,  3,  0, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 1, 10,  6,  1,  6,  0,  0,  6,  4, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 4,  8,  6,  6,  1, 10,  6,  8,  1,  1,  8,  3, -1,  0,  0,  0},
	std::array<int, 16>{ 9,  1,  2,  9,  2,  4,  4,  2,  6, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  8,  3,  9,  1,  2,  9,  2,  4,  4,  2,  6, -1,  0,  0,  0},
	std::array<int, 16>{ 0,  2,  6,  0,  6,  4, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 3,  4,  8,  3,  2,  4,  4,  2,  6, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 4, 10,  6,  4,  9, 10,  2,  3, 11, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 8,  2,  0,  8, 11,  2,  4,  9, 10,  4, 10,  6, -1,  0,  0,  0},
	std::array<int, 16>{ 2,  3, 11,  1, 10,  6,  1,  6,  0,  0,  6,  4, -1,  0,  0,  0},
	std::array<int, 16>{ 8, 11,  2,  8,  2,  4,  4,  2,  1,  4,  1, 10,  4, 10,  6, -1},
	std::array<int, 16>{ 3, 11,  1,  1,  4,  9, 11,  4,  1, 11,  6,  4, -1,  0,  0,  0},
	std::array<int, 16>{ 6,  4,  9,  6,  9, 11, 11,  9,  1, 11,  1,  0, 11,  0,  8, -1},
	std::array<int, 16>{11,  0,  3, 11,  6,  0,  0,  6,  4, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 8, 11,  6,  8,  6,  4, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 6,  7,  8,  6,  8, 10, 10,  8,  9, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 3,  0,  7,  7, 10,  6,  0, 10,  7,  0,  9, 10, -1,  0,  0,  0},
	std::array<int, 16>{ 1, 10,  6,  1,  6,  7,  1,  7,  0,  0,  7,  8, -1,  0,  0,  0},
	std::array<int, 16>{ 6,  1, 10,  6,  7,  1,  1,  7,  3, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 9,  1,  8,  8,  6,  7,  8,  1,  6,  6,  1,  2, -1,  0,  0,  0},
	std::array<int, 16>{ 7,  3,  0,  7,  0,  6,  6,  0,  9,  6,  9,  1,  6,  1,  2, -1},
	std::array<int, 16>{ 8,  6,  7,  8,  0,  6,  6,  0,  2, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 2,  6,  7,  2,  7,  3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{11,  2,  3,  6,  7,  8,  6,  8, 10, 10,  8,  9, -1,  0,  0,  0},
	std::array<int, 16>{ 9, 10,  6,  9,  6,  0,  0,  6,  7,  0,  7, 11,  0, 11,  2, -1},
	std::array<int, 16>{ 3, 11,  2,  0,  7,  8,  0,  1,  7,  7,  1,  6,  6,  1, 10, -1},
	std::array<int, 16>{ 6,  7, 10,  7,  1, 10,  7,  2,  1,  7, 11,  2, -1,  0,  0,  0},
	std::array<int, 16>{ 1,  3, 11,  1, 11,  9,  9, 11,  6,  9,  6,  7,  9,  7,  8, -1},
	std::array<int, 16>{ 6,  7, 11,  9,  1,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 8,  0,  7,  0,  6,  7,  0, 11,  6,  0,  3, 11, -1,  0,  0,  0},
	std::array<int, 16>{ 6,  7, 11, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 6, 11,  7, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 3,  0,  8, 11,  7,  6, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 6, 11,  7,  9,  0,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 1,  8,  3,  1,  9,  8,  7,  6, 11, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{11,  7,  6,  2, 10,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 1,  2, 10,  0,  8,  3, 11,  7,  6, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 9,  2, 10,  9,  0,  2, 11,  7,  6, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{11,  7,  6,  3,  2, 10,  3, 10,  8,  8, 10,  9, -1,  0,  0,  0},
	std::array<int, 16>{ 2,  7,  6,  2,  3,  7, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 8,  7,  6,  8,  6,  0,  0,  6,  2, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 7,  2,  3,  7,  6,  2,  1,  9,  0, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 8,  7,  9,  9,  2,  1,  9,  7,  2,  2,  7,  6, -1,  0,  0,  0},
	std::array<int, 16>{ 6, 10,  1,  6,  1,  7,  7,  1,  3, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 6, 10,  1,  6,  1,  0,  6,  0,  7,  7,  0,  8, -1,  0,  0,  0},
	std::array<int, 16>{ 7,  6,  3,  3,  9,  0,  6,  9,  3,  6, 10,  9, -1,  0,  0,  0},
	std::array<int, 16>{ 6,  8,  7,  6, 10,  8,  8, 10,  9, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 8,  6, 11,  8,  4,  6, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{11,  3,  0, 11,  0,  6,  6,  0,  4, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 6,  8,  4,  6, 11,  8,  0,  1,  9, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 1,  9,  3,  3,  6, 11,  9,  6,  3,  9,  4,  6, -1,  0,  0,  0},
	std::array<int, 16>{ 8,  6, 11,  8,  4,  6, 10,  1,  2, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 2, 10,  1, 11,  3,  0, 11,  0,  6,  6,  0,  4, -1,  0,  0,  0},
	std::array<int, 16>{11,  4,  6, 11,  8,  4,  2, 10,  9,  2,  9,  0, -1,  0,  0,  0},
	std::array<int, 16>{ 4,  6, 11,  4, 11,  9,  9, 11,  3,  9,  3,  2,  9,  2, 10, -1},
	std::array<int, 16>{ 3,  8,  4,  3,  4,  2,  2,  4,  6, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 2,  0,  4,  2,  4,  6, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  1,  9,  3,  8,  4,  3,  4,  2,  2,  4,  6, -1,  0,  0,  0},
	std::array<int, 16>{ 9,  2,  1,  9,  4,  2,  2,  4,  6, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 6, 10,  4,  4,  3,  8,  4, 10,  3,  3, 10,  1, -1,  0,  0,  0},
	std::array<int, 16>{ 1,  6, 10,  1,  0,  6,  6,  0,  4, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{10,  9,  0, 10,  0,  6,  6,  0,  3,  6,  3,  8,  6,  8,  4, -1},
	std::array<int, 16>{10,  9,  4, 10,  4,  6, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 6, 11,  7,  5,  4,  9, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  8,  3,  9,  5,  4,  7,  6, 11, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  5,  4,  0,  1,  5,  6, 11,  7, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 7,  6, 11,  4,  8,  3,  4,  3,  5,  5,  3,  1, -1,  0,  0,  0},
	std::array<int, 16>{ 2, 10,  1, 11,  7,  6,  5,  4,  9, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  8,  3,  1,  2, 10,  4,  9,  5, 11,  7,  6, -1,  0,  0,  0},
	std::array<int, 16>{ 6, 11,  7, 10,  5,  4, 10,  4,  2,  2,  4,  0, -1,  0,  0,  0},
	std::array<int, 16>{ 6, 11,  7,  5,  2, 10,  5,  4,  2,  2,  4,  3,  3,  4,  8, -1},
	std::array<int, 16>{ 2,  7,  6,  2,  3,  7,  4,  9,  5, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 4,  9,  5,  8,  7,  6,  8,  6,  0,  0,  6,  2, -1,  0,  0,  0},
	std::array<int, 16>{ 3,  6,  2,  3,  7,  6,  0,  1,  5,  0,  5,  4, -1,  0,  0,  0},
	std::array<int, 16>{ 1,  5,  4,  1,  4,  2,  2,  4,  8,  2,  8,  7,  2,  7,  6, -1},
	std::array<int, 16>{ 5,  4,  9,  6, 10,  1,  6,  1,  7,  7,  1,  3, -1,  0,  0,  0},
	std::array<int, 16>{ 4,  9,  5,  7,  0,  8,  7,  6,  0,  0,  6,  1,  1,  6, 10, -1},
	std::array<int, 16>{ 3,  7,  6,  3,  6,  0,  0,  6, 10,  0, 10,  5,  0,  5,  4, -1},
	std::array<int, 16>{ 4,  8,  5,  8, 10,  5,  8,  6, 10,  8,  7,  6, -1,  0,  0,  0},
	std::array<int, 16>{ 5,  6, 11,  5, 11,  9,  9, 11,  8, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  9,  5,  0,  5,  6,  0,  6,  3,  3,  6, 11, -1,  0,  0,  0},
	std::array<int, 16>{ 8,  0, 11, 11,  5,  6, 11,  0,  5,  5,  0,  1, -1,  0,  0,  0},
	std::array<int, 16>{11,  5,  6, 11,  3,  5,  5,  3,  1, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{10,  1,  2,  5,  6, 11,  5, 11,  9,  9, 11,  8, -1,  0,  0,  0},
	std::array<int, 16>{ 2, 10,  1,  3,  6, 11,  3,  0,  6,  6,  0,  5,  5,  0,  9, -1},
	std::array<int, 16>{ 0,  2, 10,  0, 10,  8,  8, 10,  5,  8,  5,  6,  8,  6, 11, -1},
	std::array<int, 16>{11,  3,  6,  3,  5,  6,  3, 10,  5,  3,  2, 10, -1,  0,  0,  0},
	std::array<int, 16>{ 2,  3,  6,  6,  9,  5,  3,  9,  6,  3,  8,  9, -1,  0,  0,  0},
	std::array<int, 16>{ 5,  0,  9,  5,  6,  0,  0,  6,  2, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 6,  2,  3,  6,  3,  5,  5,  3,  8,  5,  8,  0,  5,  0,  1, -1},
	std::array<int, 16>{ 6,  2,  1,  6,  1,  5, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 8,  9,  5,  8,  5,  3,  3,  5,  6,  3,  6, 10,  3, 10,  1, -1},
	std::array<int, 16>{ 1,  0, 10,  0,  6, 10,  0,  5,  6,  0,  9,  5, -1,  0,  0,  0},
	std::array<int, 16>{ 0,  3,  8, 10,  5,  6, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{10,  5,  6, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{11,  5, 10, 11,  7,  5, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 5, 11,  7,  5, 10, 11,  3,  0,  8, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{11,  5, 10, 11,  7,  5,  9,  0,  1, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 9,  3,  1,  9,  8,  3,  5, 10, 11,  5, 11,  7, -1,  0,  0,  0},
	std::array<int, 16>{ 2, 11,  7,  2,  7,  1,  1,  7,  5, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 3,  0,  8,  2, 11,  7,  2,  7,  1,  1,  7,  5, -1,  0,  0,  0},
	std::array<int, 16>{ 2, 11,  0,  0,  5,  9,  0, 11,  5,  5, 11,  7, -1,  0,  0,  0},
	std::array<int, 16>{ 9,  8,  3,  9,  3,  5,  5,  3,  2,  5,  2, 11,  5, 11,  7, -1},
	std::array<int, 16>{10,  2,  3, 10,  3,  5,  5,  3,  7, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 5, 10,  7,  7,  0,  8, 10,  0,  7, 10,  2,  0, -1,  0,  0,  0},
	std::array<int, 16>{ 1,  9,  0, 10,  2,  3, 10,  3,  5,  5,  3,  7, -1,  0,  0,  0},
	std::array<int, 16>{ 7,  5, 10,  7, 10,  8,  8, 10,  2,  8,  2,  1,  8,  1,  9, -1},
	std::array<int, 16>{ 7,  5,  1,  7,  1,  3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 8,  1,  0,  8,  7,  1,  1,  7,  5, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  5,  9,  0,  3,  5,  5,  3,  7, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 7,  5,  9,  7,  9,  8, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 4,  5, 10,  4, 10,  8,  8, 10, 11, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{11,  3, 10, 10,  4,  5, 10,  3,  4,  4,  3,  0, -1,  0,  0,  0},
	std::array<int, 16>{ 9,  0,  1,  4,  5, 10,  4, 10,  8,  8, 10, 11, -1,  0,  0,  0},
	std::array<int, 16>{ 3,  1,  9,  3,  9, 11, 11,  9,  4, 11,  4,  5, 11,  5, 10, -1},
	std::array<int, 16>{ 8,  4, 11, 11,  1,  2,  4,  1, 11,  4,  5,  1, -1,  0,  0,  0},
	std::array<int, 16>{ 5,  1,  2,  5,  2,  4,  4,  2, 11,  4, 11,  3,  4,  3,  0, -1},
	std::array<int, 16>{11,  8,  4, 11,  4,  2,  2,  4,  5,  2,  5,  9,  2,  9,  0, -1},
	std::array<int, 16>{ 2, 11,  3,  5,  9,  4, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 4,  5, 10,  4, 10,  2,  4,  2,  8,  8,  2,  3, -1,  0,  0,  0},
	std::array<int, 16>{10,  4,  5, 10,  2,  4,  4,  2,  0, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  1,  9,  8,  2,  3,  8,  4,  2,  2,  4, 10, 10,  4,  5, -1},
	std::array<int, 16>{10,  2,  5,  2,  4,  5,  2,  9,  4,  2,  1,  9, -1,  0,  0,  0},
	std::array<int, 16>{ 4,  3,  8,  4,  5,  3,  3,  5,  1, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  4,  5,  0,  5,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  3,  9,  3,  5,  9,  3,  4,  5,  3,  8,  4, -1,  0,  0,  0},
	std::array<int, 16>{ 4,  5,  9, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 7,  4,  9,  7,  9, 11, 11,  9, 10, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 8,  3,  0,  7,  4,  9,  7,  9, 11, 11,  9, 10, -1,  0,  0,  0},
	std::array<int, 16>{ 0,  1,  4,  4, 11,  7,  1, 11,  4,  1, 10, 11, -1,  0,  0,  0},
	std::array<int, 16>{10, 11,  7, 10,  7,  1,  1,  7,  4,  1,  4,  8,  1,  8,  3, -1},
	std::array<int, 16>{ 2, 11,  7,  2,  7,  4,  2,  4,  1,  1,  4,  9, -1,  0,  0,  0},
	std::array<int, 16>{ 0,  8,  3,  1,  4,  9,  1,  2,  4,  4,  2,  7,  7,  2, 11, -1},
	std::array<int, 16>{ 7,  2, 11,  7,  4,  2,  2,  4,  0, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 7,  4, 11,  4,  2, 11,  4,  3,  2,  4,  8,  3, -1,  0,  0,  0},
	std::array<int, 16>{ 7,  4,  3,  3, 10,  2,  3,  4, 10, 10,  4,  9, -1,  0,  0,  0},
	std::array<int, 16>{ 2,  0,  8,  2,  8, 10, 10,  8,  7, 10,  7,  4, 10,  4,  9, -1},
	std::array<int, 16>{ 4,  0,  1,  4,  1,  7,  7,  1, 10,  7, 10,  2,  7,  2,  3, -1},
	std::array<int, 16>{ 4,  8,  7,  1, 10,  2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 9,  7,  4,  9,  1,  7,  7,  1,  3, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 8,  7,  0,  7,  1,  0,  7,  9,  1,  7,  4,  9, -1,  0,  0,  0},
	std::array<int, 16>{ 4,  0,  3,  4,  3,  7, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 4,  8,  7, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 8,  9, 10,  8, 10, 11, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0, 11,  3,  0,  9, 11, 11,  9, 10, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 1,  8,  0,  1, 10,  8,  8, 10, 11, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 3,  1, 10,  3, 10, 11, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 2,  9,  1,  2, 11,  9,  9, 11,  8, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  9,  3,  9, 11,  3,  9,  2, 11,  9,  1,  2, -1,  0,  0,  0},
	std::array<int, 16>{11,  8,  0, 11,  0,  2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 2, 11,  3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 3, 10,  2,  3,  8, 10, 10,  8,  9, -1,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 9, 10,  2,  9,  2,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 3,  8,  2,  8, 10,  2,  8,  1, 10,  8,  0,  1, -1,  0,  0,  0},
	std::array<int, 16>{ 2,  1, 10, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 8,  9,  1,  8,  1,  3, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 1,  0,  9, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{ 0,  3,  8, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
	std::array<int, 16>{-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
};

const std::array<std::array<int, 2>, 12> edge_vertices = {
	std::array<int, 2>{0, 1},
	std::array<int, 2>{1, 2},
	std::array<int, 2>{2, 3},
	std::array<int, 2>{3, 0},
	std::array<int, 2>{4, 5},
	std::array<int, 2>{6, 5},
	std::array<int, 2>{6, 7},
	std::array<int, 2>{7, 4},
	std::array<int, 2>{0, 4},
	std::array<int, 2>{1, 5},
	std::array<int, 2>{2, 6},
	std::array<int, 2>{3, 7},
};

using vec3sz = std::array<size_t, 3>;
using vec3f = std::array<float, 3>;
using vec2f = std::array<float, 2>;
using vec3i = std::array<int, 3>;

const std::array<vec3i, 8> index_to_vertex = {
	vec3i{0, 0, 0},
	vec3i{1, 0, 0},
	vec3i{1, 1, 0},
	vec3i{0, 1, 0},
	vec3i{0, 0, 1},
	vec3i{1, 0, 1},
	vec3i{1, 1, 1},
	vec3i{0, 1, 1},
};

// Compute the vertex values of the cell given the ID of its bottom vertex
void compute_vertex_values(const std::vector<uint8_t> &volume, const vec3sz &dims, const vec3sz &cell,
		std::array<float, 8> &values)
{
	for (size_t i = 0; i < index_to_vertex.size(); ++i) {
		const auto &v = index_to_vertex[i];
		// We want to swap the order we go when on the top of the cube,
		// due to how the indices are labeled in the paper.
		size_t voxel = ((cell[2] + v[2]) * dims[1] + cell[1] + v[1]) * dims[0] + cell[0] + v[0];
		values[i] = volume[voxel];
	};
}

vec3f lerp_verts(const vec3i &va, const vec3i &vb, const float fa, const float fb, const float isoval) {
	float t = 0;
	if (std::abs(fa - fb) < 0.0001) {
		t = 0.0;
	} else {
		t = (isoval - fa) / (fb - fa);
	}
	return vec3f{va[0] + t * (vb[0] - va[0]),
		va[1] + t * (vb[1] - va[1]),
		va[2] + t * (vb[2] - va[2])};
}

// Serial marching cubes.
// Run the Marching Cubes algorithm on the volume to compute
// the isosurface at the desired value. The volume is assumed
// Dims should give the [x, y, z] dimensions of the volume
void marching_cubes(const std::vector<uint8_t> &volume, const vec3sz &dims,
		const float isovalue, std::vector<vec3f> &vertices)
{	
	size_t total_active = 0;
	std::array<float, 8> vertex_values;
	for (size_t k = 0; k < dims[2] - 1; ++k) {
		for (size_t j = 0; j < dims[1] - 1; ++j) {
			for (size_t i = 0; i < dims[0] - 1; ++i) {
				compute_vertex_values(volume, dims, {i, j, k}, vertex_values);
				size_t index = 0;
				for (size_t v = 0; v < 8; ++v) {
					if (vertex_values[v] <= isovalue) {
						index |= 1 << v;
					}
				}

				/* The cube vertex and edge indices for base rotation:
				 *
				 *      v7------e6------v6
				 *     / |              /|
				 *   e11 |            e10|
				 *   /   e7           /  |
				 *  /    |           /   e5
				 *  v3------e2-------v2  |
				 *  |    |           |   |
				 *  |   v4------e4---|---v5
				 *  e3  /           e1   /
				 *  |  e8            |  e9
				 *  | /              | /    y z
				 *  |/               |/     |/
				 *  v0------e0-------v1     O--x
				 */

				bool made_vert = false;
				// The triangle table gives us the mapping from index to actual
				// triangles to return for this configuration
				for (size_t t = 0; tri_table[index][t] != -1; ++t) {
					const int v0 = edge_vertices[tri_table[index][t]][0];
					const int v1 = edge_vertices[tri_table[index][t]][1];

					vec3f v = lerp_verts(index_to_vertex[v0], index_to_vertex[v1],
						vertex_values[v0], vertex_values[v1], isovalue);

					vertices.push_back({v[0] + i + 0.5f, v[1] + j + 0.5f, v[2] + k + 0.5f});
					made_vert = true;
				}
				if (made_vert) {
					++total_active;
				}
			}
		}
	}
}

inline vec3sz voxel_id_to_voxel(const size_t id, const vec3sz &dims) {
	return vec3sz{id % (dims[0] - 1),
		(id / (dims[0] - 1)) % (dims[1] - 1),
		id / ((dims[0] - 1) * (dims[1] - 1))
	};
}

bool voxel_is_active(const std::vector<uint8_t> &volume, const vec3sz &dims,
		const float isovalue, const size_t voxel_id)
{
	const vec3sz voxel = voxel_id_to_voxel(voxel_id, dims);
	std::array<float, 8> vertex_values;
	compute_vertex_values(volume, dims, voxel, vertex_values);

	size_t index = 0;
	for (size_t v = 0; v < 8; ++v) {
		if (vertex_values[v] <= isovalue) {
			index |= 1 << v;
		}
	}
	return index != 0 && index != tri_table.size() - 1;
}

void compute_num_verts(const std::vector<uint8_t> &volume, const vec3sz &dims,
		const float isovalue, const size_t voxel_id, const size_t active_id,
		std::vector<uint32_t> &num_verts)
{	
	const vec3sz voxel = voxel_id_to_voxel(voxel_id, dims);

	std::array<float, 8> vertex_values;
	compute_vertex_values(volume, dims, voxel, vertex_values);

	size_t index = 0;
	for (size_t v = 0; v < 8; ++v) {
		if (vertex_values[v] <= isovalue) {
			index |= 1 << v;
		}
	}

	uint32_t nverts = 0;
	// The triangle table gives us the mapping from index to actual
	// triangles to return for this configuration
	for (size_t t = 0; tri_table[index][t] != -1; ++t) {
		++nverts;
	}
	num_verts[active_id] = nverts;
}

void generate_vertices(const std::vector<uint8_t> &volume, const vec3sz &dims,
		const float isovalue, const size_t voxel_id, const size_t active_id,
		const std::vector<uint32_t> &offsets, std::vector<vec3f> &vertices)
{	
	const vec3sz voxel = voxel_id_to_voxel(voxel_id, dims);
	const uint32_t vertex_offset = offsets[active_id];

	std::array<float, 8> vertex_values;
	compute_vertex_values(volume, dims, voxel, vertex_values);
	size_t index = 0;
	for (size_t v = 0; v < 8; ++v) {
		if (vertex_values[v] <= isovalue) {
			index |= 1 << v;
		}
	}

	// The triangle table gives us the mapping from index to actual
	// triangles to return for this configuration
	for (size_t t = 0; tri_table[index][t] != -1; ++t) {
		const int v0 = edge_vertices[tri_table[index][t]][0];
		const int v1 = edge_vertices[tri_table[index][t]][1];

		vec3f v = lerp_verts(index_to_vertex[v0], index_to_vertex[v1],
				vertex_values[v0], vertex_values[v1], isovalue);
		vertices[vertex_offset + t] = {
			v[0] + voxel[0] + 0.5f, v[1] + voxel[1] + 0.5f, v[2] + voxel[2] + 0.5f
		};
	}
}

void data_parallel_marching_cubes(const std::vector<uint8_t> &volume, const vec3sz &dims,
		const float isovalue, std::vector<vec3f> &vertices)
{
	// Determine which voxels will generate vertices. The last layer of voxels don't output verts
	const size_t voxels_to_process = (dims[0] - 1) * (dims[1] - 1) * (dims[2] - 1);
	std::vector<uint32_t> voxel_active(voxels_to_process, 0);
	tbb::parallel_for(size_t(0), voxels_to_process,
		[&](const size_t v) {
			if (voxel_is_active(volume, dims, isovalue, v)) {
				voxel_active[v] = 1;
			}
		});

	// Exclusive scan to compute total number of active voxels and the offsets to write their ID
	// to in the compaction
	std::vector<uint32_t> offsets;
	const uint32_t total_active = exclusive_scan(voxel_active, uint32_t(0), offsets, std::plus<uint32_t>{});

	// Compact the active voxel IDs
	std::vector<size_t> active_voxels(total_active, 0);
	tbb::parallel_for(size_t(0), voxels_to_process,
		[&](const size_t v) {
			if (voxel_active[v]) {
				active_voxels[offsets[v]] = v;
			}
		});

	// Free voxel_active memory
	voxel_active = std::vector<uint32_t>();

	// Determine the number of vertices generated by each active voxel
	std::vector<uint32_t> num_verts(total_active, 0);
	tbb::parallel_for(size_t(0), num_verts.size(),
		[&](const size_t v) {
			compute_num_verts(volume, dims, isovalue, active_voxels[v], v, num_verts);
		});

	// Next we perform an exclusive scan to compute the offsets to write the output
	// vertices to for each voxel, and the total number of vertices we'll generate
	offsets.clear();
	const uint32_t total_verts = exclusive_scan(num_verts, uint32_t(0), offsets, std::plus<uint32_t>{});

	// Now we can compute the vertices for each voxel in parallel and write to the corresponding offsets
	vertices.resize(total_verts);
	tbb::parallel_for(size_t(0), num_verts.size(),
		[&](const size_t v) {
			generate_vertices(volume, dims, isovalue, active_voxels[v], v, offsets, vertices);
		});

}

int main(int argc, char **argv) {
	std::vector<std::string> args(argv, argv + argc);
	std::string fname;
	std::string output;
	vec3sz dims = {0};
	float isovalue = 0;
    int benchmark_iters = 1;
    vec2f bench_range = {0};
	bool serial = false;
	for (int i = 1; i < argc; ++i) {
		if (args[i] == "-f") {
			fname = argv[++i];
		} else if (args[i] == "-dims") {
			dims[0] = std::atoi(argv[++i]);
			dims[1] = std::atoi(argv[++i]);
			dims[2] = std::atoi(argv[++i]);
		} else if (args[i] == "-iso") {
			isovalue = std::atof(argv[++i]);
        } else if (args[i] == "-bench") {
            bench_range[0] = std::atof(argv[++i]);
            bench_range[1] = std::atof(argv[++i]);
            benchmark_iters = 100;
		} else if (args[i] == "-o") {
			output = args[++i];
		} else if (args[i] == "-serial") {
			serial = true;
		}
	}

	const size_t n_voxels = dims[0] * dims[1] * dims[2];
	if (fname.empty() || n_voxels == 0) {
		std::cout << "Usage: " << args[0] << " -f <file.raw> -dims <x> <y> <z> -iso <v>\n"
			<< "\tThe volume file must contain uint8_t row major data\n";
	}

	std::ifstream fin(fname.c_str(), std::ios::binary);
	std::vector<uint8_t> volume(n_voxels, 0);
	fin.read(reinterpret_cast<char*>(volume.data()), volume.size());

    size_t total_time = 0;
    float value_range = bench_range[1] - bench_range[0];
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> distrib;
	std::vector<vec3f> vertices;
    for (size_t i = 0; i < benchmark_iters; ++i) {
        vertices.clear();
        if (benchmark_iters != 1) {
            isovalue = bench_range[0] + value_range * distrib(rng);
            std::cout << "isovalue: " << isovalue << "\n";
        }

        auto start = high_resolution_clock::now();

        if (serial) {
            marching_cubes(volume, dims, isovalue, vertices);
        } else {
            data_parallel_marching_cubes(volume, dims, isovalue, vertices);
        }

        auto end = high_resolution_clock::now();
        auto dur = duration_cast<milliseconds>(end - start).count();
        total_time += dur;

        std::cout << "Isosurface with " << vertices.size() / 3 << " triangles computed in "
            << dur << "ms " << (serial ? "(serial)\n" : "(parallel)\n");
    }
    std::cout << "Average compute time: " << static_cast<float>(total_time) / benchmark_iters << "ms\n"; 

	if (!output.empty()) {
		std::ofstream fout(output.c_str());
		fout << "# Isosurface of " << fname << " at isovalue " << isovalue * 255.f << "\n";
		for (const auto &v : vertices) {
			fout << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
		}

		// Every three pairs of vertices forms a face
		for (size_t i = 0; i < vertices.size(); i += 3) {
			fout << "f " << i + 1 << " " << i + 2 << " " << i + 3 << "\n";
		}
	}

	return 0;
}

