# CUDAGameOfLife
Conway's Game of Life in CUDA/C++

Usage: cudagol [OPTION]... [FILE [Y] [X]] [FILE [Y] [X]]...

FILE: Plaintext-format cells file (see [ConwayLife wiki entry](https://conwaylife.com/wiki/Plaintext))

[Y] [X]: Optional offsets from top-left corner to load the file into

[OPTION]: any of -b -w -p

* -b: Wait for user to press any key to continue, or q to quit
* -w [MICROSECONDS]: Microseconds per frame draw
* -p [PRINTER]: Choose drawing method:
    * c[C]: Live cells represented by C, dead cells blank. C is # if not defined.
    * color: Live cells green with number of neighbors in their input displayed, dead cells red with number of neighbors in their input displayed as 'a'+neighbors
    * fancy: position-based ascii characters for live cells, space for dead cells
    * neighbors: number of neighbors in input for live cells, space for dead cells

Special thanks to [the ConwayLife wiki](https://conwaylife.com/wiki/)
