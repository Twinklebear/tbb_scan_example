cmake_minimum_required(VERSION 3.5)
project(tbb_scan_example)

if (NOT WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
else()
	add_definitions(-DNOMINMAX)
endif()

find_package(Threads REQUIRED)
find_package(TBB REQUIRED)

add_executable(scan_example main.cpp)
target_link_libraries(scan_example PUBLIC TBB::tbb)

