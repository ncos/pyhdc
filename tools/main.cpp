#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include <iostream>
#include <fstream>

#define BYTE_WIDTH 4
#define BIT_WIDTH (8 * BYTE_WIDTH)
#define VECTOR_WIDTH (BIT_WIDTH * 4)
#define DEPTH_X 10
#define DEPTH_Y 10


std::string print_vector(uint32_t vec[VECTOR_WIDTH]) {
    std::string ret;
    ret = "{" + std::to_string(vec[0]);
    for (unsigned long int i = 1; i < VECTOR_WIDTH; ++i) {
        ret += ", " + std::to_string(vec[i]);
    }
    ret += "}";
    return ret;
}


std::string print_vector_tree(uint32_t vec[VECTOR_WIDTH]) {
    std::string ret;
    ret = "{{" + std::to_string(vec[0] / BIT_WIDTH) + ", " + std::to_string(vec[0] % BIT_WIDTH) + "}";
    for (unsigned long int i = 1; i < VECTOR_WIDTH; ++i) {
        ret += ", {" + std::to_string(vec[i] / BIT_WIDTH) + ", " + std::to_string(vec[i] % BIT_WIDTH) + "}";
    }
    ret += "}";
    return ret;
}


void permute_self(uint32_t vec[VECTOR_WIDTH]) {
    uint32_t tmp[VECTOR_WIDTH];
    memcpy(tmp, vec, VECTOR_WIDTH * sizeof(*vec));

    for (unsigned long int i = 0; i < VECTOR_WIDTH; ++i) {
        vec[tmp[i]] = tmp[i];
    }
}


std::string print_permutation_array(std::string name, uint32_t _seed[VECTOR_WIDTH], uint32_t depth) {
    uint32_t seed[VECTOR_WIDTH];
    memcpy(seed, _seed, VECTOR_WIDTH * sizeof(*_seed));
    
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


std::string print_permutation_array_tree(std::string name, uint32_t _seed[VECTOR_WIDTH], uint32_t depth) {
    uint32_t seed[VECTOR_WIDTH];
    memcpy(seed, _seed, VECTOR_WIDTH * sizeof(*_seed));
    
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


int main() {
    uint32_t Seed_x[VECTOR_WIDTH] = {1, 2, 3, 4, 5, 6, 7, 0};
    uint32_t Seed_y[VECTOR_WIDTH] = {1, 2, 3, 5, 4, 6, 7, 0};

    for (unsigned long int i = 0; i < VECTOR_WIDTH; ++i) {
        Seed_x[i] = (i + 1) % VECTOR_WIDTH;
        Seed_y[i] = (i + 10) % VECTOR_WIDTH;
    }


    std::ofstream ofs;
    ofs.open ("permutations.h", std::ofstream::out);

    ofs << "#ifndef PERMUTATIONS_H" << std::endl;
    ofs << "#define PERMUTATIONS_H" << std::endl;
    ofs << std::endl;
    ofs << "#define VECTOR_WIDTH " << VECTOR_WIDTH << std::endl;
    ofs << "#define DEPTH_X " << DEPTH_X << std::endl;
    ofs << "#define DEPTH_Y " << DEPTH_Y << std::endl;
    ofs << "#define BYTE_WIDTH " << BYTE_WIDTH << std::endl;
    ofs << "#define BIT_WIDTH "  << BIT_WIDTH << std::endl;
    ofs << std::endl;

    ofs << print_permutation_array("__Px", Seed_x, DEPTH_X) << std::endl;
    ofs << print_permutation_array_tree("Px", Seed_x, DEPTH_X) << std::endl;
    ofs << std::endl;
    //ofs << print_permutation_array("__Py", Seed_y, DEPTH_Y) << std::endl;
    //ofs << print_permutation_array_tree("Py", Seed_y, DEPTH_Y) << std::endl;

    ofs << std::endl;
    ofs << "#endif // PERMUTATIONS_H" << std::endl;
    ofs.close();
    return 0;
};
