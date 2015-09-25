import numpy
import gf2n
import pytest

@pytest.mark.parametrize("generator, value, degree", [
    (0x101, 0x8f, 8),
    (0x1234, 0x321, 12),
    (0xb, 0x3, 3),
])
def test_instantiate(generator, value, degree):
    test = gf2n.gf2n(generator, value)
    assert(test.generator == generator)
    assert(test.value == value)
    assert(test.degree == degree)


@pytest.mark.parametrize("generator, v1, v2, out", [
    (0x101, 0x8f, 0x8f, 0x0),
    (0x101, 0x12, 0x32, 0x20),
])
def test_add(generator, v1, v2, out):
    test1 = gf2n.gf2n(generator, v1)
    test2 = gf2n.gf2n(generator, v2)
    res = test1 + test2

    assert(res.generator == test1.generator)
    assert(res.value == out)
    assert(res.degree == test1.degree)


@pytest.mark.parametrize("generator, v1, v2, out", [
    (0x101, 0x43, 0x32, 0x71),
    (0x101, 0x12, 0x44, 0x56),
    (0x101, 0x1e, 0x8f, 0x91),
])
def test_sub(generator, v1, v2, out):
    test1 = gf2n.gf2n(generator, v1)
    test2 = gf2n.gf2n(generator, v2)
    res = test1 - test2

    assert(res.generator == test1.generator)
    assert(res.value == out)
    assert(res.degree == test1.degree)

@pytest.mark.parametrize("generator, v1, v2, out", [
    (0xb, 0x3, 0x3, 0x5),
    (0xb, 0x5, 0x4, 0x2),
    (0xb, 0x7, 0x2, 0x5),
])
def test_mult(generator, v1, v2, out):
    test1 = gf2n.gf2n(generator, v1)
    test2 = gf2n.gf2n(generator, v2)
    res = test1 * test2

    assert(res.generator == test1.generator)
    assert(res.value == out)
    assert(res.degree == test1.degree)


