import sys
import itertools
import numpy as np
from scipy.stats import itemfreq
from collections import defaultdict


def read_ali(ali_filename):
    """ Read multiple sequence alignment in jones format,
    i.e. each row represents one aligned protein sequence.

    @param  ali_filename    path to alignment file
    @return ali             numpy array of sequences in numerical representation.
    """

    with open(ali_filename) as ali_file:
        for i, seq in enumerate(ali_file):
            pass
    
    # number of sequences
    B = i+1
    
    with open(ali_filename) as ali_file:
        seq = ali_file.readline().strip()

    # sequence length
    N = len(seq)

    ali = np.zeros((B,N))
    aa_dict = {'R':1, 'H':2, 'K':3, 'D':4, 'E':5, 'S':6, 'T':7,\
            'N':8, 'Q':9, 'C':10, 'G':11, 'P':12, 'A':13, 'I':14,\
            'L':15, 'M':16, 'F':17, 'W':18, 'Y':19, 'V':20, '-':21}

    with open(ali_filename) as ali_file:
        for i, seq in enumerate(ali_file):
            seq = seq.strip()
            ali[i,:] = [aa_dict[aa] for aa in seq]
        
    return ali


def get_freq_single(ali):
    """ Calculate column-wise frequencies of amino acids:
    f_i(k) = 1/B * sum_B(delta(pos_i(b), k)), where B=#seqs, k=1-21.

    @param  ali         np array of sequences in numerical representation
    @return f_single    np array of column-wise frequencies
    """
    B, N = ali.shape
    f_single = np.zeros((21,N))
    for i, col in enumerate(ali.T):
        f_pre = dict(itemfreq(col))
        for k in range(1,22):
            if k in f_pre.keys():
                f_single[k-1,i] = 1./B * f_pre[k]
    return f_single



def get_freq_pair(ali):
    """ Calculate column-wise frequencies of amino acids for all pairs of columns:
    f_ij(k,l) = 1/B * sum(delta(pos_i(b), k) * delta(pos_j(b), l))

    @param  ali         np array of sequences in numerical representation
    @return f_pair      np array of column-wise frequencies for all pairs of columns
    """
    B, N = ali.shape
    f_pair = np.zeros((21*N,21*N))
    for i, col_i in enumerate(ali.T):
        print '%d/%d' % (i,N)
        for j, col_j in enumerate(ali.T):
            for (k, l) in itertools.product(range(1,22), repeat=2):
                sum_b = sum([1 for b in range(B) if col_i[b] == k and col_j[b] == l])
                f_pair[21*i+k-1, 21*j+l-1] = 1./B * float(sum_b)
    return f_pair


def cov_mat(f_single, f_pair):
    """ Calculate covariance matrix from frequencies:
    c_ij(k,l) = f_ij(k,l) - f_i(k)*f_j(l)

    @param  f_single    np array of column-wise frequencies
    @param  f_pair      np array of column-wise frequencies for all pairs of columns
    @return C           np array containing the covariance matrix
    """
    N = f_pair.shape[0]/21
    C = np.zeros((21*N,21*N))
    for (i, j) in itertools.product(range(N), repeat=2):
        for (k, l) in itertools.product(range(1,22), repeat=2):
            C[21*i+k-1, 21*j+l-1] = f_pair[21*i+k-1, 21*j+l-1] - f_single[k-1,i] * f_single[l-1,j]

    return C


def main(ali_filename):

    """
    f_i(k) = 1/B * sum_B(delta(pos_i(b), k)), where B=#seqs, k=1-21
    f_ij(k,l) = 1/B * sum(delta(pos_i(b), k) * delta(pos_j(b), l))

    c_ij(k,l) = f_ij(k,l) - f_i(k)*f_j(l)
    """

    ali = read_ali(ali_filename)
    print ali
    f_single = get_freq_single(ali)
    print f_single
    print f_single.shape
    f_pair = get_freq_pair(ali)
    print f_pair
    print f_pair.shape
    C = cov_mat(f_single, f_pair)
    print C
    print C.shape
    np.savetxt("%s.covmat.csv" % ali_filename, C, delimiter=",")


if __name__ == "__main__":

    ali_filename = sys.argv[1]
    main(ali_filename)
