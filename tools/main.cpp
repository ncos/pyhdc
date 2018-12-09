#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>


#define BIT_WIDTH 32


std::string print_vector(std::vector<uint32_t> &vec) {
    std::string ret;
    uint32_t VECTOR_WIDTH = vec.size();
    ret = "{" + std::to_string(vec[0]);
    for (unsigned long int i = 1; i < VECTOR_WIDTH; ++i) {
        ret += ", " + std::to_string(vec[i]);
    }
    ret += "}";
    return ret;
}


std::string print_vector_tree(std::vector<uint32_t> &vec) {
    std::string ret;
    uint32_t VECTOR_WIDTH = vec.size();
    ret = "{{" + std::to_string(vec[0] / BIT_WIDTH) + ", " + std::to_string(vec[0] % BIT_WIDTH) + "}";
    for (unsigned long int i = 1; i < VECTOR_WIDTH; ++i) {
        ret += ", {" + std::to_string(vec[i] / BIT_WIDTH) + ", " + std::to_string(vec[i] % BIT_WIDTH) + "}";
    }
    ret += "}";
    return ret;
}


void permute_self(std::vector<uint32_t> &vec) {
    std::vector<uint32_t> tmp;
    tmp = vec;
    uint32_t VECTOR_WIDTH = vec.size();

    for (unsigned long int i = 0; i < VECTOR_WIDTH; ++i) {
        vec[i] = tmp[tmp[i]];
    }
}


std::string print_permutation_array(std::string name, std::vector<uint32_t> &_seed, uint32_t depth) {
    std::vector<uint32_t> seed;
    seed = _seed;
    uint32_t VECTOR_WIDTH = seed.size();

    std::string ret;
    ret = "static const uint32_t " + name + "[" + std::to_string(depth) + "][" + std::to_string(VECTOR_WIDTH) + "] = {\n";
    ret += "    " + print_vector(seed) + "\n";

    for (unsigned long int i = 1; i < depth; ++i) {
        permute_self(seed);
        ret += "   ," + print_vector(seed) + "\n";
    }

    ret += "};";
    return ret;
}


std::string print_permutation_array_tree(std::string name, std::vector<uint32_t> &_seed, uint32_t depth) {
    std::vector<uint32_t> seed;
    seed = _seed;
    uint32_t VECTOR_WIDTH = seed.size();
    
    std::string ret;
    ret = "static const uint8_t " + name + "[" + std::to_string(depth) + "][" + std::to_string(VECTOR_WIDTH) + "][2] = {\n";
    ret += "    " + print_vector_tree(seed) + "\n";

    for (unsigned long int i = 1; i < depth; ++i) {
        permute_self(seed);
        ret += "   ," + print_vector_tree(seed) + "\n";
    }

    ret += "};";
    return ret;
}


void generate(std::string dir, uint32_t DEPTH_X, uint32_t DEPTH_Y, uint8_t nbins) {
    uint32_t VECTOR_WIDTH = BIT_WIDTH * nbins; // 1 - 255

    std::vector<uint32_t> Seed_x, Seed_y;
    for (unsigned long int i = 0; i < VECTOR_WIDTH; ++i) {
        Seed_x.push_back((i + 1) % VECTOR_WIDTH);
        Seed_y.push_back((i + 10) % VECTOR_WIDTH);
    }
    
    // Shuffle the permutation vectors
    std::srand(42);
    std::random_shuffle(Seed_x.begin(), Seed_x.end());
    std::random_shuffle(Seed_y.begin(), Seed_y.end());

    std::ofstream ofs;
    ofs.open (dir + "/permutations_" + std::to_string(VECTOR_WIDTH) + ".h", std::ofstream::out);

    ofs << "#ifndef PERMUTATIONS_H" << std::endl;
    ofs << "#define PERMUTATIONS_H" << std::endl;
    ofs << std::endl;
    ofs << "#define VECTOR_WIDTH " << VECTOR_WIDTH << std::endl;
    ofs << "#define DEPTH_X " << DEPTH_X << std::endl;
    ofs << "#define DEPTH_Y " << DEPTH_Y << std::endl;
    ofs << "#define BIT_WIDTH "  << BIT_WIDTH << std::endl;
    ofs << std::endl;

    //ofs << print_permutation_array("__Px", Seed_x, DEPTH_X) << std::endl;
    ofs << print_permutation_array_tree("Px", Seed_x, DEPTH_X) << std::endl;
    ofs << std::endl;
    //ofs << print_permutation_array("__Py", Seed_y, DEPTH_Y) << std::endl;
    ofs << print_permutation_array_tree("Py", Seed_y, DEPTH_Y) << std::endl;

    ofs << std::endl;
    ofs << "#endif // PERMUTATIONS_H" << std::endl;
    ofs.close();
}


int main() {
    generate("../lib", 350, 350, 4);
    generate("../lib", 350, 350, 100);
    generate("../lib", 350, 350, 255);
    return 0;
};
