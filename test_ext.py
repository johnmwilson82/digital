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


@pytest.mark.parametrize("generator, value, nonzero", [
    (0x101, 0x8f, True),
    (0x1234, 0x0, False),
    (0xb, 0x3, True),
    (0xb, 0x0, False),
])

def test_nonzero(generator, value, nonzero):
    test = gf2n.gf2n(generator, value)
    assert(test.generator == generator)
    assert(test.value == value)
    if nonzero:
        assert(test)
    else:
        assert(not test)


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
    (0x101, 0x43, 0x32, 0x71),
    (0x101, 0x12, 0x44, 0x56),
    (0x101, 0x1e, 0x8f, 0x91),
])
def test_xor(generator, v1, v2, out):
    test1 = gf2n.gf2n(generator, v1)
    test2 = gf2n.gf2n(generator, v2)
    res = test1 ^ test2

    assert(res.generator == test1.generator)
    assert(res.value == out)
    assert(res.degree == test1.degree)


@pytest.mark.parametrize("generator, v1, v2, out", [
    (0x101, 0x43, 0x32, 0x73),
    (0x101, 0x12, 0x44, 0x56),
    (0x101, 0x1e, 0x8f, 0x9f),
])
def test_or(generator, v1, v2, out):
    test1 = gf2n.gf2n(generator, v1)
    test2 = gf2n.gf2n(generator, v2)
    res = test1 | test2

    assert(res.generator == test1.generator)
    assert(res.value == out)
    assert(res.degree == test1.degree)


@pytest.mark.parametrize("generator, v1, v2, out", [
    (0x101, 0x43, 0x32, 0x02),
    (0x101, 0x12, 0x44, 0x00),
    (0x101, 0x1e, 0x8f, 0x0e),
])
def test_and(generator, v1, v2, out):
    test1 = gf2n.gf2n(generator, v1)
    test2 = gf2n.gf2n(generator, v2)
    res = test1 & test2

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


@pytest.mark.parametrize("generator, v1, shift, out", [
    (0x101, 0x18, 2, 0x60),
    (0x101, 0x18, 4, 0x80),
    (0x101, 0x11, 1, 0x22),
    (0x101, 0x03, 1, 0x06),
])
def test_lshift(generator, v1, shift, out):
    test1 = gf2n.gf2n(generator, v1)
    res = test1 << shift

    assert(res.generator == test1.generator)
    assert(res.value == out)
    assert(res.degree == test1.degree)


@pytest.mark.parametrize("generator, v1, shift, out", [
    (0x101, 0x18, 2, 0x06),
    (0x101, 0x18, 4, 0x01),
    (0x101, 0x11, 1, 0x08),
    (0x101, 0x03, 1, 0x01),
])
def test_rshift(generator, v1, shift, out):
    test1 = gf2n.gf2n(generator, v1)
    res = test1 >> shift

    assert(res.generator == test1.generator)
    assert(res.value == out)
    assert(res.degree == test1.degree)


@pytest.mark.parametrize("generator, v1, mod, out", [
    (0x101, 0x18, 0x5, 0x04),
    (0x101, 0x18, 0x10, 0x08),
    (0x101, 0x36, 0xb, 0xa),
])
def test_mod(generator, v1, mod, out):
    test1 = gf2n.gf2n(generator, v1)
    res = test1 % mod

    assert(res.generator == test1.generator)
    assert(res.value == out)
    assert(res.degree == test1.degree)


