import numpy
import gf2n

test1 = gf2n.gf2n(0x101, 0x01)

print hex(test1.generator)
print hex(test1.value)
print test1.degree

test2 = gf2n.gf2n(0x1234, 0x321)

print hex(test2.generator)
print hex(test2.value)
print test2.degree

test3 = test1 + test2

print test3.value
