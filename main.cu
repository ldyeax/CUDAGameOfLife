/**
 * Conway's Game of Life implementation in CUDA
 * github.com/ldyeax
 * jimm@jimm.horse
 **/

// Set to 0 to let the game keep running without input,
//  at US_PER_FRAME microseconds per frame
#define BLOCKING 1

#include <chrono>
#include <thread>
#include <stdio.h>
#include <ncurses.h>
#include <unistd.h>
#include "lib.cu"
#include "spaceships.cu"

float *d_Input;
float *d_Output;
float *h_Input;
float *h_Output;

#define WIDTH 80
#define HEIGHT 24

#define US_PER_FRAME 1000000 

/**
 * Size of all our input/output arrays
 **/
const size_t MDSIZE = sizeof(float)*WIDTH*HEIGHT;

/**
 * Taking block index as position on the board,
 *  compute the game of life to output from input.
 * 1 thread per block only
 **/
__global__ void CalculateCell(
	float *input, 
	float *output)
{
	int nNeighbors = 0;
	for (int y = blockIdx.y - 1; y <= blockIdx.y + 1; y++) {
		for (int x = blockIdx.x - 1; x <= blockIdx.x + 1; x++) {
			if (y < 0 || x < 0) {
				continue;
			}
			if (y == blockIdx.y && x == blockIdx.x) {
				continue;
			}
			if (MDGET(input, y, x) > 0.5f) {
				nNeighbors++;
			}
		}
	}
	if (MDGET(input, blockIdx.y, blockIdx.x)) {
		if (nNeighbors < 2 || nNeighbors > 3) {
			MDGET(output, blockIdx.y, blockIdx.x) = 0;
		} else {
			MDGET(output, blockIdx.y, blockIdx.x) = 1;
		}
	} else {
		if (nNeighbors == 3) {
			MDGET(output, blockIdx.y, blockIdx.x) = 1;
		} else {
			MDGET(output, blockIdx.y, blockIdx.x) = 0;
		}
	}
}

int main(int argc, char** argv)
{
	// Allocate input/output on both host and device
	h_Input = (float*)malloc(MDSIZE);
	if (!h_Input) {
		printf("Failed to malloc h_Input\n");
		return 1;
	}
	h_Output = (float*)malloc(MDSIZE);
	if (!h_Output) {
		printf("h_Output malloc failed\n");
		return 1;
	}

	gpuErrChk(cudaMalloc(&d_Input, MDSIZE));
	gpuErrChk(cudaMalloc(&d_Output, MDSIZE)); 
/*
	// Put hwss spaceship in center
	int partHeight = 6;
	int partWidth = 7;

	int yInputStart = HEIGHT/2 - 3;
	int xInputStart = WIDTH/2 - 4;

	for (int i = 0; i < partHeight; i++) {
		for (int j = 0; j < partWidth; j++) {
			MDGET(	h_Input,
				yInputStart + i,
				xInputStart + j
			) = spaceships::hwss[i][j];
		}
	}
*/

	for (int i = 1; i < argc; i++) {
		char* filename = argv[i];
		int yOffset = 0;
		int xOffset = 0;
		bool gotY = false;
		if (i + 1 < argc) {
			gotY = testNumeric(argv[i + 1], &yOffset);
		}
		if (gotY) {
			i++;
		}
		if (gotY && i + 1 < argc) {
			if (testNumeric(argv[i + 1], &xOffset)) {
				i++;
			}
		}
		readCellsFile(filename, yOffset, xOffset);
	}

	// Initialize input and output on GPU from host
	gpuErrChk(cudaMemcpy(
		d_Input, 
		h_Input, 
		MDSIZE, 
		cudaMemcpyHostToDevice));
	gpuErrChk(cudaMemcpy(
		d_Output, 
		h_Output, 
		MDSIZE,
		cudaMemcpyHostToDevice));
	

	// One CUDA block per cell
	dim3 numBlocks(WIDTH, HEIGHT);
	// Each block only needs 1 thread
	dim3 threadsPerBlock(1);

	ncurses_setup();
#if !BLOCKING
	auto clock = std::chrono::high_resolution_clock::now();
#endif

	float *swp1 = h_Output;
	h_Output = h_Input;
	print_hOutput();
	h_Output = swp1;

#if BLOCKING
	if (getch() == 'q') {
		ncurses_cleanup();
		return 0;
	}
#endif

	while (true) {
		// Process game of life on all cells simultaneously
		CalculateCell<<<numBlocks, threadsPerBlock>>>(
			d_Input, d_Output);
				
		// Get output from GPU
		gpuErrChkF(
			cudaMemcpy(
				h_Output, 
				d_Output, 
				MDSIZE,
				cudaMemcpyDeviceToHost), 
			ncurses_cleanup);
#if !BLOCKING
		auto now = std::chrono::high_resolution_clock::now();
		auto duration = now - clock;
		clock = now;
		auto microseconds = 
			std::chrono::duration_cast
			<std::chrono::microseconds>
			(duration).count();
		if (microseconds < US_PER_FRAME) {
			std::this_thread::sleep_for(
				std::chrono::microseconds(
					US_PER_FRAME - microseconds));
		}
#endif
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
