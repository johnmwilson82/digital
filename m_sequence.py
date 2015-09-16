import utils
from scipy import signal
import matplotlib.pyplot as plt

# Output common polynomials for generating m-sequences of given degree
def poly(degree):
    poly_dict = {
        1: [1, 0],
        2: [2, 1, 0],
        3: [3, 1, 0],
        4: [4, 1, 0],
        5: [5, 2, 0],
        6: [6, 1, 0],
        7: [7, 1, 0],
        8: [8, 7, 2, 1, 0],
        9: [9, 4, 0],
        10: [10, 3, 0],
        11: [11, 2, 0],
        12: [12, 6, 4, 1, 0]
        }
    return poly_dict[degree]

# Generate an m-sequence with a given polynomial as described above
def gen_m_sequence(poly, init=0x2):
    # convert poly into fsr taps
    taps = sum([2 ** n for n in poly])
    mask = (2 ** (1+poly[0])) - 1
    bins = init
    done = 0
    output = []
    while not done:
        output.append(1 if bins & 1 else 0)
        bins = ((bins >> 1) +
                ((utils.count_bits(taps & bins) % 2) << (poly[0]-1))) & mask
        if bins == init or bins == 0:
            done = 1
    return output

# Quick testing
if __name__ == '__main__':
    output = gen_m_sequence(poly(12))
    corr = signal.correlate(output, output+output+output, mode='same')
    plt.plot(corr)
    plt.show()

