Covariance Matrix from Alignments in CUDA
=========================================

CUDA implementation of the initial steps of direct coupling analysis
(DCA). Given a multiple sequence alignment, the tool calculates
residue frequencies and the covariance matrix as described in
"[Jones et al. 2012](http://www.ncbi.nlm.nih.gov/pubmed/22101153)" and "[Ekeberg et
al. 2013](http://arxiv.org/pdf/1211.1281.pdf)".


Problem
-------

One possibility to predict the structure
of a protein is to first predict contacts between pairs of amino acids
within the protein. These contacts are then be used to guide the
structure prediction process. Amino acid contact prediction requires
large multiple sequence alignments of protein families. One subproblem
in contact prediction is to compute a covariance matrix from the
alignment. This is done by first calculating the frequencies of each
amino acid at each position in the alignment and the frequencies of
every possible amino acid pair at every possible pair of positions
(equation (4) in Ekeberg et al. 2013). Based on these observed frequencies
the covariance matrix is calculated according to equation (5) in Ekeberg
et al. 2013.


Implementation
--------------

The calculations are implemented in three versions using C++ and CUDA: serially on the CPU, using un-optimized kernels in CUDA, and an optimized version. 
