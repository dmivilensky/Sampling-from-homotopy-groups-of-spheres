#include <vector>
#include <utility>
#include <iostream>

#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

std::vector<std::pair<int, int>> split_word(const py::array_t<int>& word, std::size_t word_idx, int generator) {
    std::vector<std::pair<int, int>> result;
    result.reserve(word.shape(1));

    for (py::ssize_t i = 0; i < word.shape(1) && word.at(word_idx, i) != 0; ++i) {
        if (std::abs(word.at(word_idx, i)) == std::abs(generator)) {
            result.emplace_back(i, std::copysign(1, word.at(word_idx, i)));
        }
    }

    return result;
}

template <typename T>
std::vector<std::pair<int, T>> prefix_sums(
    const std::vector<std::pair<int, int>> &splits,
    const std::vector<std::pair<int, T>> &p_sums
) {
    std::vector<std::pair<int, T>> result;
    result.reserve(splits.size());

    int i_split = 0, i_sums = 0;
    for (; i_split < splits.size(); ++i_split) {

        if (p_sums.size() == 0) {
            result.emplace_back(splits[i_split].first, 0);
            continue;
        }

        int idx = splits[i_split].first + (splits[i_split].second == -1 ? 1: 0);
        int d_sums = 0;

        while (i_sums + d_sums < p_sums.size() && p_sums[i_sums + d_sums].first < idx) {
            d_sums += 1;
        }
        if (d_sums > 0) {
            i_sums += d_sums - 1;
        }
        if (p_sums[i_sums].first >= idx) {
            result.emplace_back(splits[i_split].first, 0);
        } else {
            int sign = splits[i_split].second;
            long long current = p_sums[i_sums].second;

            if (result.size() == 0) {
                result.emplace_back(splits[i_split].first, sign * current);
            } else {
                long long previous = result.back().second;
                result.emplace_back(splits[i_split].first, previous + sign * current);
            }
        }
    }
    return result;
}


template <typename T>
void magnus_coefficients(
    const py::array_t<int>& word,
    std::size_t word_idx,
    std::size_t generators_num,
    std::size_t modulo,
    py::array_t<T> coefficients
) {
    std::size_t coef_idx = 0;
    coefficients.mutable_at(word_idx, coef_idx++) = 1.;

    std::vector<std::vector<std::pair<int, int>>> splits;
    splits.reserve(generators_num);
    for (int i = 1; i <= generators_num; ++i) {
        splits.push_back(split_word(word, word_idx, i));
    }

    std::vector<std::vector<std::pair<int, T>>> previous_prefixes = {{{-1, 1}}};
    for (std::size_t len = 0; len < modulo; ++len) {
        std::vector<std::vector<std::pair<int, T>>> current_prefixes;
        current_prefixes.reserve(previous_prefixes.size() * generators_num);
        for (const auto& split: splits) {
            for (const auto& prefix : previous_prefixes) {
                current_prefixes.push_back(prefix_sums(split, prefix));
            }
        }
        previous_prefixes.swap(current_prefixes);
        for (const auto& prefix: previous_prefixes) {
            coefficients.mutable_at(word_idx, coef_idx++) = prefix.size() == 0 ? 0 : prefix.back().second;
        }
    }
}

template<typename T> void derivative(
    const py::array_t<int>& word,
    std::size_t word_idx,
    std::size_t generators_num,
    const py::array_t<int>& wrt,
    py::array_t<T> coefficients
) {
    std::vector<std::vector<std::pair<int, int>>> splits;
    splits.reserve(generators_num);
    for (int i = 1; i <= generators_num; ++i) {
        splits.push_back(split_word(word, word_idx, i));
    }

    for (std::size_t wrt_idx = 0; wrt_idx < wrt.shape(0); ++wrt_idx) {
        std::vector<std::pair<int, T>> prefixes = {{-1, 1}};
        for (std::size_t i = 0; i < wrt.shape(1); ++i) {
            std::size_t gen_idx = wrt.at(wrt_idx, wrt.shape(1) - 1 - i) - 1;
            prefixes = prefix_sums(splits[gen_idx], prefixes);
        }
        coefficients.mutable_at(word_idx, wrt_idx) = prefixes.size() == 0 ? 0 : prefixes.back().second;
    }
}


template<typename T> py::array_t<T> magnus_coefficients(py::array_t<int> words, std::size_t generators_num, std::size_t modulo) {
    // auto words_buffer = words.request();

    py::ssize_t dimension = static_cast<py::ssize_t>(std::pow(generators_num, modulo + 1)) / (generators_num - 1);
    auto result = py::array_t<T>((words.shape(0) * dimension));
    result.resize({words.shape(0), dimension});
    // auto result_buffer = result.request();

    for (ssize_t word_idx = 0; word_idx < words.shape(0); ++word_idx) {
        magnus_coefficients(words, word_idx, generators_num, modulo, result);
    }

    return result;
}

template <typename T> py::array_t<T> derivative(py::array_t<int> words, std::size_t generators_num, py::array_t<int> wrt) {
    auto result = py::array_t<T>((words.shape(0) * wrt.shape(0)));
    result.resize({words.shape(0), wrt.shape(0)});

    for (ssize_t word_idx = 0; word_idx < words.shape(0); ++word_idx) {
        derivative(words, word_idx, generators_num, wrt, result);
    }

    return result;
}

PYBIND11_MODULE(derivatives, m) {
    m.def(
        "magnus_coefficients",
        static_cast<py::array_t<long long>(*)(py::array_t<int>, std::size_t, std::size_t)>(&magnus_coefficients),
        "Computes coefficients of the Magnus embedding"
    );
    m.def(
        "derivative",
        static_cast<py::array_t<long long>(*)(py::array_t<int>, std::size_t, py::array_t<int>)>(&derivative),
        "Computes the fox derivative for all words with respect to `wrt`"
    );
}


