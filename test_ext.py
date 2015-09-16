import numpy
import gf2n

test = gf2n.gf2n(16, numpy.array([1, 2, 3, 4]))

print test.value

test = gf2n.gf2n(16, "4321")

print test.value
