# Quick bit counting function
def count_bits(n):
  n = (n & 0x5555555555555555) + ((n & 0xAAAAAAAAAAAAAAAA) >> 1)
  n = (n & 0x3333333333333333) + ((n & 0xCCCCCCCCCCCCCCCC) >> 2)
  n = (n & 0x0F0F0F0F0F0F0F0F) + ((n & 0xF0F0F0F0F0F0F0F0) >> 4)
  n = (n & 0x00FF00FF00FF00FF) + ((n & 0xFF00FF00FF00FF00) >> 8)
  n = (n & 0x0000FFFF0000FFFF) + ((n & 0xFFFF0000FFFF0000) >> 16)
  n = (n & 0x00000000FFFFFFFF) + ((n & 0xFFFFFFFF00000000) >> 32)
  return n

def get_highest_set_bit(a):
    i = -1
    while a:
        i += 1
        a -= a & (1 << i)

    return i

def find_prime_factors(a):
    ret = []
    def next_prime_factor(aa):
        for bb in range(2, aa):
            cc = aa / bb
            if cc * bb == aa:
                return bb

    while 1:
        b = next_prime_factor(a)
        if not b:
            ret.append(a)
            return ret

        ret.append(b)
        a /= b
        if a == 1:
            return ret

def gf2_mod(a, mod):
    if mod == 0:
        return 0

    while 1:
        i = get_highest_set_bit(a)
        j = get_highest_set_bit(mod)

        diff = i - j

        if diff < 0:
            return a
        a ^= (mod << diff)

def gf2_div_with_rem(a, mod):
    if mod == 0:
        return (0, 0)

    quot = 0

    while 1:
        i = get_highest_set_bit(a)
        j = get_highest_set_bit(mod)

        diff = i - j

        if diff < 0:
            return (quot, a)

        a ^= (mod << diff)
        quot += 1 << diff


def gf2_gcd(a, b):
    # Using the Euclidean algorithm
    (quot, rem) = gf2_div_with_rem(a, b)
    while rem != 0:
        a = b
        b = rem
        (quot, rem) = gf2_div_with_rem(a, b)
    return b


if __name__ == "__main__":
    #tt = find_prime_factors(99)
    bbb = gf2_gcd(0b1111100, 0b110110)
    aaa = gf2_mod(0b1111100, 0b10)
    print bin(bbb)
    print bin(aaa)
    import pdb
    pdb.set_trace()

