#!/usr/bin/python3

import pyhdc

print ("Max order on x:", pyhdc.get_max_order('x'))
print ("Max order on y:", pyhdc.get_max_order('y'))
print ("Vector width:", pyhdc.get_vector_width(), "bits")
print ()

a = pyhdc.LBV()
a.rand()
print ("Vector a:")
print (a)
print ()

print ("P0:")
print (pyhdc.permutation_to_str('x', 0))
print ()
print ("P1:")
print (pyhdc.permutation_to_str('x', 1))
print ()

b = pyhdc.LBV()
b.xor(a) # copy a to b


print ("a vs a permuted with P0:")
print (a)
a.permute('x', 0)
print (a)
print ()
print ("a vs a permuted with P1:")
print (b)
b.permute('x', 1)
print (b)
print ()
print ("a vs a permuted with P0 three times:")
print (b)
for i in range(3):
    b.permute('x', 0)
print (b)

# ===== Performance =====
import time

print ()
num = 5000000
print ("Running", num, "XORs (same):")
c = pyhdc.LBV()
c.rand();
start = time.time()
for i in range(num):
    c.xor(a)

end = time.time()
print ((end - start) / float(num), "sec per xor")

print ()
num = 1000000
print ("Running", num, "permutations (same):")
start = time.time()
for i in range(num):
    b.permute('x', 0)

end = time.time()
print ((end - start) / float(num), "sec per permutation")

print ()
print ("Running", num, "permutations (sequential):")
start = time.time()
maxorder = pyhdc.get_max_order('x')
for i in range(num):
    b.permute('x', i % maxorder)

end = time.time()
print ((end - start) / float(num), "sec per permutation")
