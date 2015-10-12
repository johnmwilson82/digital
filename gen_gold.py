import gf2n
import utils
import gen_primitive_polys as gpp
import numpy as np
import scipy as sc
import m_sequence
import matplotlib.pyplot as plt
from scipy import signal

def gen_preferred_pair(poly, q):
    """
    A) n is odd or mod(n,4)=2

    B) q is odd and either q=2k+1 or q=2^(2k)-2k+1 for an integer k.

    C) The greatest common divisor of n and k satisfies the following conditions:
        gcd(n,k)=1, when n is odd
        gcd(n,k)=2, when mod(n,4)=2
    """
    #n = 11
    #q = 5
    #k = 2
    d1 = m_sequence.gen_m_sequence(poly)
    d2 = np.array([d1[(i*q) % len(d1)] for i in xrange(len(d1))])

    return (d1, d2)

def gen_gold_codes(num, poly, q):
    d1, d2 = gen_preferred_pair(poly, q)

    ret = [d1, d2, [i % 2 for i in d1+d2]]
    if num <= 3:
        return ret[:num]

    def shift_seq(seq, shift):
        return np.concatenate((seq[shift:], seq[-shift:]))

    for i in xrange(num - 3):
        ret.append([i % 2 for i in d1 + shift_seq(d2, i+1)])

    return ret

if __name__ == "__main__":
    ppp = gen_gold_codes(10, 0b100100111011, 5)
    for i in range(len(ppp)):
        ppp[i] = [(p - 0.5) * 2.0 for p in ppp[i]]

    corr1 = signal.correlate(ppp[0], np.concatenate((ppp[0],ppp[0],ppp[0])), mode='same')
    corr2 = signal.correlate(ppp[0], np.concatenate((ppp[1],ppp[1],ppp[1])), mode='same')
    corr3 = signal.correlate(ppp[6], np.concatenate((ppp[6],ppp[6],ppp[6])), mode='same')
    corr4 = signal.correlate(ppp[4], np.concatenate((ppp[8],ppp[8],ppp[8])), mode='same')
    plt.plot(corr1)
    plt.plot(corr2)
    plt.plot(corr3)
    plt.plot(corr4)
    plt.show()

    import pdb
    pdb.set_trace()

