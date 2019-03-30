#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <vector>



float P(int nbits, int id, std::vector<float> &prob_list) {
    float p = prob_list[id];
    if (id == prob_list.size() - 1) {
        if (nbits > 1) return 0.0;
        if (nbits == 1) return p;
        if (nbits <= 0) return 1.0;
    }
    return (1 - p) * P(nbits, id + 1, prob_list) + p * P(nbits - 1, id + 1, prob_list);
}

uint64_t product(uint64_t begin, uint64_t end) {
    uint64_t ret = 1;
    for (uint64_t i = begin; i <= end; ++i)
        ret *= i;

    return ret;
}

double exact(int n_, int k_, float p) {
    int n = n_, k = k_;
    if (k_ > n_ - k_)
        k = n - k;

    double res = std::pow(p, k_) * std::pow(1 - p, n_ - k_);
    for (int i = 0; i < k; ++i) { 
        res *= double(n - i); 
        res /= double(i + 1); 
    }

    //std::cout << res << "\n";
    return res;
}

float Pf(int nbits, float p2, int size) { 
    float res = 0;
    float p = p2;
    for (int k = nbits; k <= size; ++k) {
        res += exact(size, k, p);
        //double C = binomial(prob_list.size() - id, k);
        //res += C * std::pow(p, k) * std::pow(1 - p, prob_list.size() - id - k);
        //res += C / std::pow(2, prob_list.size() - id);
        //std::cout << prob_list.size() - id << " " << k << " " << C << " " << std::pow(2, prob_list.size() - id) << "\n";
    }
    return res;
}

float P_fast(int nbits, float p1, float p2, int size) {
    float p = p1;
    return (1 - p) * Pf(nbits, p2, size - 1) + p * Pf(nbits - 1, p2, size - 1);
}

int main (int argc, char* argv[]) {
    int size = std::stoi(argv[1]);
    float prob = std::stof(argv[2]);
    float prob2 = std::stof(argv[3]);
    int nbits = int(ceil(float(size) / 2.0));
    std::cout << size << "; " << prob << "| " << nbits << std::endl;

    std::vector<float> prob_list;
    prob_list.push_back(prob);
    for (int i = 0; i < size - 1; ++i)
        prob_list.push_back(prob2);

    //std::cout << P_fast(nbits, prob, prob2, prob_list.size()) << std::endl;
    //std::cout << P(nbits, 0, prob_list) << std::endl;
    
    for (int i = 0; i < 500; ++i) {
        size = 2 * i + 1;
        nbits = int(ceil(float(size) / 2.0));
        std::cout << size;
        
        for (float p1 = 0; p1 <= prob; p1 += 0.1)
            std::cout << " " << P_fast(nbits, p1, prob2, size);

        std::cout << std::endl;
    }
    
    return 0;
};
