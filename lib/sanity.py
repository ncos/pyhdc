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

a.permute('x', 0)
b.permute('x', 1)

print ("a permuted with P0:")
print (a)
print ()
print ("a permuted with P1:")
print (b)
