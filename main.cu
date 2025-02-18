/**
 * Conway's Game of Life implementation in CUDA
 * github.com/ldyeax
 * jimm@jimm.horse
 **/

#include <stdio.h>
#include <ncurses.h>
#include <math.h>
#include "lib.cuh"
#include "lib.cu"
#include "spaceships.cu"

namespace CUDAGameOfLife
{
	/**
	 * Taking block index as position on the board,
	 *  compute the game of life to output from input.
	 * 1 thread per block only
	 **/
	__global__ void CalculateCell(
		float *input, 
		float *output,
		int width,
		int height)
	{
		int nNeighbors = 0;

		int startY = blockIdx.y ? blockIdx.y - 1 : 0;
		int startX = blockIdx.x ? blockIdx.x - 1: 0;
		int endY = blockIdx.y + 1;
		if (endY >= height) {
			endY--;
		}
		int endX = blockIdx.x + 1;
		if (endX >= width) {
			endX--;
		}

		for (int y = startY; y <= endY; y++) {
			for (int x = startX; x <= endX; x++) {
				if (y == blockIdx.y && x == blockIdx.x) {
					continue;
				}
				if (MDGET(input, y, x) > 0.5f) {
					nNeighbors++;
				}
			}
		}
		// If this cell was alive..
		if (MDGET(input, blockIdx.y, blockIdx.x) > 0.5f) {
			// Die by loneliness or overpopulation
			if (nNeighbors < 2 || nNeighbors > 3) {
				MDGET(output, blockIdx.y, blockIdx.x)
					= copysign(nNeighbors, -1.0f);
			// Stay alive if 2 or 3 neighbors
			} else {
				MDGET(output, blockIdx.y, blockIdx.x) 
					= copysign(nNeighbors, 1.0f);
			}
		// If this cell was dead..
		} else {
			// Come alive if there are exactly 3 neighbors
			if (nNeighbors == 3) {
				MDGET(output, blockIdx.y, blockIdx.x) 
					= copysign(nNeighbors, 1.0f);
			// Stay dead without exactly 3 neighbors
			} else {
				MDGET(output, blockIdx.y, blockIdx.x) 
					= copysign(nNeighbors, -1.0f);
			}
		}
	}

	int main(int argc, char** argv)
	{
		if (argc == 1) {
			printHelp();
			return 1;
		}
		initWidthHeight();
		
		// Allocate input/output on both host and device
		h_Input = (float*)malloc(mdsize);
		if (!h_Input) {
			printf("Failed to malloc h_Input\n");
			return 1;
		}
		h_Output = (float*)malloc(mdsize);
		if (!h_Output) {
			printf("h_Output malloc failed\n");
			return 1;
		}

		gpuErrChk(cudaMalloc(&d_Input, mdsize));
		gpuErrChk(cudaMalloc(&d_Output, mdsize)); 

		if (int e = parseArguments(argc, argv)) {
			return e;
		}

		// Initialize input and output on GPU from host
		gpuErrChk(cudaMemcpy(
			d_Input, 
			h_Input, 
			mdsize, 
			cudaMemcpyHostToDevice));
		gpuErrChk(cudaMemcpy(
			d_Output, 
			h_Output, 
			mdsize,
			cudaMemcpyHostToDevice));

		// One CUDA block per cell
		dim3 numBlocks(width, height);
		// Each block only needs 1 thread
		dim3 threadsPerBlock(1);

		ncurses_setup();
		
		if (!blocking) {
			startClock();
		}
		
		// Swap input/output and print the initial screen
		float *swp1 = h_Output;
		h_Output = h_Input;
		print_hOutput();
		h_Output = swp1;

		if (blocking && getch() == 'q') {
			ncurses_cleanup();
			return 0;
		}

		while (true) {
			// Process game of life on all cells simultaneously
			CalculateCell<<<numBlocks, threadsPerBlock>>>(
				d_Input, d_Output, width, height);
					
			// Get output from GPU
			gpuErrChkF(
				cudaMemcpy(
					h_Output, 
					d_Output, 
					mdsize,
					cudaMemcpyDeviceToHost), 
				ncurses_cleanup);
			if (!blocking) {
				waitForNextFrame();
			}

			int alive = print_hOutput();
			char keypress = getch();
			if (alive == 0 || keypress == 'q') {
				break;
			};

			// Swap inputs and outputs for next iteration
			// The kernel function writes over all of output,
			//  so we don't need to pre-clear it
			float* tmp;

			tmp = d_Input;
			d_Input = d_Output;
			d_Output = tmp;

			tmp = h_Input;
			h_Input = h_Output;
			h_Output = tmp;
		}
		ncurses_cleanup();
		return 0;
	}
}
int main (int argc, char** argv)
{
	return CUDAGameOfLife::main(argc, argv);
}
