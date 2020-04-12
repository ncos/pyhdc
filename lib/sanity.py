#!/usr/bin/python3

import pyhdc

import time, random

print ("Max order on x:", pyhdc.get_max_order('x'))
print ("Max order on y:", pyhdc.get_max_order('y'))
print ("Vector width:", pyhdc.get_vector_width(), "bits")
print ()

a = pyhdc.LBV()
a.rand() # Get a random vector
print ("Vector a:")
print (a)
print ()

# There are 'x' and 'y' permutations; the seed (first one) is generated separately
# and independently for both; the second value is 'order' - how many times the
# seed permutation should be permuted with itself. All permutations are precomputed, so
# only up to 'pyhdc.get_max_order' orders can be used.
print ("P0:")
print (pyhdc.permutation_to_str('x', 0))
print ()
print ("P1:")
print (pyhdc.permutation_to_str('x', 1))
print ()




def test_csum(n=101, mode='1'):
    print ("Csum test")

    v = [pyhdc.LBV() for i in range(n)]
    for i in range(n):
        v[i].rand()

    res = pyhdc.csum(v, mode)

    res_ref = pyhdc.LBV()
    th = len(v) / 2.0
    for i in range(pyhdc.get_vector_width()):
        cnt = 0
        for v_ in v:
            if (v_.get_bit(i)):
                cnt += 1
        if (abs(cnt - th) < 1e-3):
            if (mode == '0'):
                continue
            if (mode == '1'):
                res_ref.flip(i)
                continue
            if (mode == 'r' and random.choice([True, False])):
                res_ref.flip(i)
                continue

        if (cnt > th):
            res_ref.flip(i)

    res_ref.xor(res);
    print ("Passed:", res_ref.is_zero())
    print ()
    return res_ref.is_zero()


def test_permute(v_, axis, order, times):
    v = pyhdc.LBV()
    v.xor(v_) # copy v_ to v

    print ("v permuted with P" + axis + str(order), times, "time(s)")
    print (v)
    for i in range(times):
        v.permute(axis, order)
    print (v)
    for i in range(times):
        v.inv_permute(axis, order)
    print (v)

    # Check if the same after inverse permutation
    v.xor(v_);
    print ("Passed:", v.is_zero())
    print ()
    return v.is_zero()



def test_bitmanip(v_):
    v = pyhdc.LBV()
    v.xor(v_) # copy v_ to v

    print ("Read bit test")
    ref_str = str(v)
    print (ref_str)
    test_str = ""
    for i in range(pyhdc.get_vector_width()):
        if (v.get_bit(i)):
            test_str += '1'
        else:
            test_str += '0'

        if ((i + 1) % 32 == 0):
            test_str += '_'
    print (test_str)
    t1_result = (test_str == ref_str)
    print ("Passed:", t1_result)
    print ()

    print ("Flip bit test")
    print (v)
    for i in range(pyhdc.get_vector_width() // 2):
        v.flip(i * 2)
    print (v)
    for i in range(pyhdc.get_vector_width() // 2):
        v.flip(i * 2)
    print (v)
    for i in range(pyhdc.get_vector_width()):
        if (v.get_bit(i)):
            v.flip(i)
    print (v)
    t2_result = v.is_zero()
    print ("Passed:", t2_result)
    print ()

    print ("Count bit test")
    nbits = 0
    for i in range(pyhdc.get_vector_width()):
        choice = random.choice([True, False])
        if (choice):
            v.flip(i)
            nbits += 1
    test_nbits = v.count()
    print (v)
    t3_result = (nbits == test_nbits)
    print ("Passed:", t3_result, "true value = ",
           nbits, "result = ", test_nbits)
    return t1_result & t2_result & t3_result


passed = True
passed &= test_csum(101, '0')
passed &= test_csum(101, '1')
passed &= test_permute(a, 'x', 0, 1)
passed &= test_permute(a, 'y', 1, 1)
passed &= test_permute(a, 'x', 0, 100)
passed &= test_bitmanip(a)
print("All tests passed:", passed)

# ===== Performance =====

print ()
num = 5000000
print ("Running", num, "XORs (same):")
c = pyhdc.LBV()
c.rand()
start = time.time()
for i in range(num):
    c.xor(a)

end = time.time()
print ((end - start) / float(num), "sec per xor")

print ()
num = 100000
print ("Running", num, "permutations (same):")
a.rand()
start = time.time()
for i in range(num):
    a.permute('x', 0)

end = time.time()
print ((end - start) / float(num), "sec per permutation")

print ()
print ("Running", num, "permutations (sequential):")
a.rand()
maxorder = pyhdc.get_max_order('x')
start = time.time()
for i in range(num):
    a.permute('x', i % maxorder)

end = time.time()
print ((end - start) / float(num), "sec per permutation")

print ()
print ("Running", num, "permutations (random):")
a.rand()
original = pyhdc.LBV()
original.xor(a)

axes = ['x', 'y']

permutations = []
for i in range(num):
    axis = random.choice(axes)
    nperm = random.randint(0, pyhdc.get_max_order(axis))
    permutations.append([axis, nperm])

start = time.time()
for i in range(num):
    a.permute(permutations[i][0], permutations[i][1])

end = time.time()
print ((end - start) / float(num), "sec per permutation")

print ()
print ("Checking inverse transform...")
start = time.time()
for i in range(num - 1, -1, -1):
    a.inv_permute(permutations[i][0], permutations[i][1])

end = time.time()
a.xor(original)
print ((end - start) / float(num), "sec per (inverse) permutation")
print ("Test passed:", a.is_zero())
print ()

print ("Bit manipulation performance")
num = 5000000
a.rand()

bitseq = []
for i in range(num):
    bitseq.append(random.randint(0, pyhdc.get_vector_width() - 1))

start = time.time()
for bit in bitseq:
    a.flip(bit)

end = time.time()
print ((end - start) / float(num), "sec per bit flip")
print ()

a.rand()

bitseq = []
for i in range(num):
    bitseq.append(random.randint(0, pyhdc.get_vector_width() - 1))

start = time.time()
pos_cnt = 0
for bit in bitseq:
    if (a.get_bit(bit)):
        pos_cnt += 1
end = time.time()
print ((end - start) / float(num), "sec per bit lookup")
print ("Positive percentage:", int(pos_cnt / num * 100))
print ()

a.rand()
start = time.time()
for i in range(num):
    a.count()

end = time.time()
print ((end - start) / float(num), "sec per bit count")
print ()
