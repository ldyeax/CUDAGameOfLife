# CUDAGameOfLife
Conway's Game of Life in CUDA/C++

- Build with build.sh
- Adjust options with #DEFINE flags in main.cu
- Load a cells file like './main glider.cells 4 5' to put the contents of glider.cells at y=4 x=5
- Cells file format follows [Plaintext](https://conwaylife.com/wiki/Plaintext)

There's a bug with anything in the topmost row of cells, and the implementation might have more bugs I haven't found yet.

Special thanks to [the ConwayLife wiki](https://conwaylife.com/wiki/)
