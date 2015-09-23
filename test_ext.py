import numpy
import gf2n

test = gf2n.gf2n(16, numpy.array([1, 2, 3, 4]))

print test.value

test = gf2n.gf2n(16, "1010101010104321")

print [format(i, '02x') for i in test.value]
