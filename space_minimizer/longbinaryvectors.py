import numpy as np
import random


class LongBinaryVector:

    vector = None

    def __init__(self, length, seed):
        if length < 1:
            raise ValueError('Given vector length cannot be less than 1!')

        self.vector = np.zeros((1, length), dtype=bool)

        for i in np.nditer(self.vector, op_flags=['readwrite']):
            if not random.getrandbits(1):
                i[...] = True

    def permute(self, p):
        return self.vector[p]


def hamming_distance(x, y):
    count = 0

    for i in range(min(len(x.vector[0]), len(y.vector[0]))):
        if x.vector[0][i] != y.vector[0][i]:
            count += 1

    return count


def normalized_hamming_distance(x, y):
    count = hamming_distance(x, y)

    return count / len(x.vector)


def vector_xor(x, y):
    return np.bitwise_xor(x.vector, y.vector)


def consensus_sum(s):
    x = np.zeros((1, len(next(iter(s)))))

    for e in s:
        for i in range(len(e.vector)):
            if e.vector[0][i]:
                x[i] += 1

    return x


def consensus_vote(x, s):
    n = len(s)
    x /= n

    if n % 2 == 1:
        b = x > 0.5
    else:
        b = np.zeros((1, len(next(iter(s)))), dtype=bool)

        for i in range(len(x)):
            if x[i] > 0.5:
                b[i] = True
            elif x[i] == 0.5:
                if not random.getrandbits(1):
                    b[i] = True

    return b
