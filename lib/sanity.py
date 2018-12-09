#!/usr/bin/python3

import pyhdc

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

passed = True
passed &= test_permute(a, 'x', 0, 1)
passed &= test_permute(a, 'y', 1, 1)
passed &= test_permute(a, 'x', 0, 100)
print("All tests passed:", passed)

# ===== Performance =====
import time, random

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
