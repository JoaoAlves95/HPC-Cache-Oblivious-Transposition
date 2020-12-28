# HPC-Cache-Oblivious-Transposition
High performance cache-oblivious out-of-place matrix transposition based on a blocking scheme that follows a Hilbert space-filling curve.

The source file requires minor refactorization in order to exactly match the algorithm description presented in the article. The solutions are however equivalent.

To run this program (requires GCC compiler):

1) Download the source folder;
2) cd source;
3) make;
4) ./mtx n m, where n and m represent the number of rows and columns fo the automatically generated random matrix.

The program outputs the time it took to perform the transposition step.
