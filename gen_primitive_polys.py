import gf2n
import utils
import numpy as np

def berlekamp_factorisation_check(a):
    # Check for two or more irreducable factors
    # from http://www.diva-portal.org/smash/get/diva2:414578/FULLTEXT01.pdf
    # gcd(f(x), f'(x))
    # (a >> 1) & 0x77777777 is a quick way of differentiating a gf2 poly < 32 bits
    if utils.gf2_gcd(a, (a >> 1) & 0x77777777) != 1:
        return False

    degree = utils.get_highest_set_bit(a)
    polys = [utils.gf2_mod(1 << (2*i), a) ^ (1 << i) for i in xrange(degree)]

    for i, div in enumerate(polys[:i]):
        for j in range(len(polys[i+1:])):
            polys[j+i+1] ^= div
            if polys[j+i+1] == 0:
                # Matrix of polys is not full rank
                return False

    return True


def get_primitive_polys_gf2(deg, max_num=None, random=True):
    # From http://www.seanerikoconnor.freeservers.com/Mathematics/AbstractAlgebra/PrimitivePolynomials/theory.html
    # Step 1: Generate a polynomial
    r = (2 ** deg) - 1

    ret = []

    for poly in xrange(1 << deg, 1 << (deg + 1)):
        # Step 2 (not needed for GF2)

        # Step 3: check f(a)!=0 for 1 <= a <= p-1
        if (utils.count_bits(poly) % 2) == 0:
            continue
        # Step 4:
        if not berlekamp_factorisation_check(poly):
            continue
        # Step 5: x**r == a(mod f(x), p) for some integer a (gf(2) means a == 1)
        if utils.gf2_mod(1 << r, poly) != 1:
            continue

        # Step 6: skip

        # Step 7:
        p_k = list(set(utils.find_prime_factors(r)))
        passed = True
        for p in p_k:
            if utils.gf2_mod(1 << (r/p), poly) == 1:
                passed = False
        if not passed:
            continue

        #print poly
        ret.append(poly)
        print "Found poly = " + bin(poly)
        if max_num and len(ret) >= max_num:
            return ret

    return ret

if __name__ == "__main__":
    prims = get_primitive_polys_gf2(11)
    print [bin(c) for c in prims]
    import pdb
    pdb.set_trace()
