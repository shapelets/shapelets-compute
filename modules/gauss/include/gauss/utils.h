#ifndef GAUSS_UTILS_H
#define GAUSS_UTILS_H

#include <gauss/defines.h>
#include <gauss/dimensionality.h>

#include <iostream>

void printVectorSegment(const std::vector<algos::dimensionality::Segment> &seg,
                        std::vector<algos::dimensionality::Point> ts) {
    std::cout << "VECTORPOINT: " << seg.size() << std::endl;
    int i = 0;
    for (auto &s : seg) {
        std::cout << "POS[" << i++ << "]= (" << ts[s.first].first << ", " << ts[s.first].second << ") , "
                  << "(" << ts[s.second].first << ", " << ts[s.second].second << ")" << std::endl;
    }
    std::cout << std::endl;
}

void printVectorPoint(const std::vector<algos::dimensionality::Point> &vec) {
    std::cout << "VECTORPOINT: " << vec.size() << std::endl;
    int i = 0;
    for (auto &v : vec) {
        std::cout << "POS[" << i++ << "]= (" << v.first << ", " << v.second << ")" << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void printVector(std::vector<T> vec) {
    std::cout << "PRINTVECTOR: " << vec.size() << std::endl;
    for (auto &v : vec) {
        std::cout << v << "\t";
    }
    std::cout << std::endl;
}

template <typename T>
void printArray(T *array, int len) {
    std::cout << "PRINTARRAY: " << len << std::endl;
    for (int i = 0; i < len; i++) {
        std::cout << array[i] << "\t";
    }
    std::cout << std::endl;
}

#endif
