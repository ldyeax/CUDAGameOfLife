#include <iostream>
#include <stdio.h>
#include <fstream>
#include <chrono>
#include <thread>
#include <sys/ioctl.h>
#include <unistd.h>
#include <ncurses.h>

namespace CUDAGameOfLife
{
	// Input/output matrices for device and host
	float *d_Input;
	float *d_Output;
	float *h_Input;
	float *h_Output;

	int width = 80;
	int height = 24;

	/**
	 * Blocking (wait for keypress to continue)
	 * or non-blocking (keep simulating unattended)
	 **/
	bool blocking = false;

	/**
	 * If blocking, wait this long before showing next render
	 **/
	int usPerFrame = 25000; // 250000;

	bool useColor = true;
	
	/**
	 * Size of all our input/output arrays
	 **/
	size_t mdsize;

	void initWidthHeight()
	{
		struct winsize w;
		ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
		width = w.ws_col;
		height = w.ws_row;
		mdsize = sizeof(float)*width*height;
	}

	/**
	 * If str is numeric, set result to the integer value;
	 *  otherwise return false and do nothing
	 **/
	bool testNumeric(char* str, int* result)
	{
		int len = strlen(str);
		for (int i = 0; i < len; i++) {
			char c = str[i];
			if (c < '0' || c > '9') {
				return false;
			}
		}
		*result = atoi(str);
		return true;
	}

	/**
	 * Read .cells format file into h_input
	 **/
	int readCellsFile(char* filename, int offsetY, int offsetX)
	{
		std::ifstream file;
		file.open(filename, std::ifstream::in);
		if (!file) {
			printf("Error opening file %s\n", filename);
			return 1;
		}
		std::string line;
		int lineNumber = 0;
		while (std::getline(file, line)) {
			int linelen = line.length();
			if (!linelen) {
				continue;
			}
			if (line[0] == '!') {
				continue;
			}
			for (int i = 0; i < linelen; i++) {
				if (line[i] != '.' && line[i] != ' ') {
					MDGET(	h_Input,
						offsetY + lineNumber,
						offsetX + i) = 1;
				}
			}
			lineNumber++;
		}
		file.close();
		return 0;
	}

	void printHelp()
	{
		printf("Conway's Game of Life in CUDA\n");
		printf(" To quit: Press q\n");
		printf("Usage: cudagol [OPTION]... "
			"[FILE [Y] [X]] [FILE [Y] [X]]...\n");
		printf("FILE: Plaintext-format cells file "
			"(see https://conwaylife.com/wiki/Plaintext) \n");
		printf("[Y] [X]: Optional offsets from top-left corner "
			"to load the file into\n");
		printf("[OPTION]: any of -b -w -p\n");
		printf(" -b: Wait for user to press any key to continue,"
			" or q to quit\n");
		printf(" -w [MICROSECONDS]: "
			"Microseconds per frame draw\n");
		printf(" -p [PRINTER]: Choose drawing method: \n");
		printf("  c[C]: Live cells represented by C,"
			" dead cells blank. C is # if not defined.\n");
		printf("  color: Live cells green with number of"
			" neighbors in their input displayed,"
			" dead cells red with number of neighbors"
			" in their input displayed as 'a'+neighbors\n");
		printf("  fancy: position-based ascii characters"
			" for live cells, space for dead cells\n");
		printf("  neighbors: number of neighbors in input"
			" for live cells, space for dead cells\n");
	}

	int parseArguments(int argc, char **argv)
	{
		for (int i = 1; i < argc; i++) {
			std::string arg = argv[i];
			if (arg == "-help" || arg == "--help") {
				printHelp();
				return 1;
			}
			if (arg == "-b") {
				blocking = true;
				continue;
			}
			if (arg == "-w") {
				i++;
				if (i >= argc) {
					printHelp();
					return 1;
				}
				if (!testNumeric(argv[i], &usPerFrame)) {
					printHelp();
					return 1;
				}
				continue;
			}
			if (arg == "-p") {
				i++;
				if (i >= argc) {
					printHelp();
					return 1;
				}
				std::string printer = argv[i];
				if (printer.length() == 2 
					&& printer[0] == 'c') {
					printchar_f = printchar__char;
					printchar__char__char = printer[1];
					continue;
				}
				if (printer == "color") {
					printchar_f = printchar__color;
					useColor = true;
					continue;
				}
				if (printer == "fancy") {
					printchar_f = printchar__fancy;
					continue;
				}
				if (printer == "neighbors") {
					printchar_f = printchar__neighbors;
					continue;
				}
				printHelp();
				return 1;
			}

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
		return 0;
	}

	/**
	 * If code is not success:
	 * - run optional beforePrint function
	 * - print error, file, line
	 * - exit program if abort=true
	 **/
	inline void gpuAssert(
			cudaError_t code, 
			const char *file,
			int line,
			bool abort = true,
			void (*beforePrint)() = NULL) 
	{
		// printf("%s\n", cudaGetErrorString(code));
		if (code != cudaSuccess) {
			if (beforePrint) {
				beforePrint();
			}
			printf(
			// fprintf(
			//	stderr, 
				"GPUassert: %s %s %d\n",
				cudaGetErrorString(code),
				file,
				line);
			printf("%-10s\t%p\n", "d_Input",  d_Input);
			printf("%-10s\t%p\n", "d_Output", d_Output);
			printf("%-10s\t%p\n", "h_Input",  h_Input);
			printf("%-10s\t%p\n", "h_Output", h_Output);
			if (abort) {
				printf("abort");
				exit(code);
			}
		}
	}

	char printchar__char__char = '#';
	int printchar__char(float cell, int y, int x)
	{
		if (cell > 0.5f) {
			mvaddch(y, x, printchar__char__char);
			return 1;
		} else {
			mvaddch(y, x, ' ');
			return 0;
		}
	}
	/**
	 * Get an arbitrary ASCII char from 33 to 126,
	 *  based on deterministic modulo of input
	 **/
	int printchar__fancy(float cell, int y, int x) 
	{
		if (cell > 0.5f) {
			static int fancyMax = 126;
			static int fancyMin = 33;
			int n = y*width + x;
			n %= (fancyMax - fancyMin);
			mvaddch(y, x, n + fancyMin);
			return 1;
		} else {
			mvaddch(y, x, ' ');
			return 0;
		}
	}

	int printchar__neighbors(float cell, int y, int x)
	{
		if (cell > 0.5f) {
			mvaddch(y, x, '0' + (int)cell);
			return 1;
		} else {
			mvaddch(y, x, ' ');
			return 0;
		}
	}

	int printchar__color(float cell, int y, int x)
	{
		int cellInt = (int)cell;
		if (cellInt > 0) {
			attron(COLOR_PAIR(cellInt));
			mvaddch(y, x, '0' + cellInt);
			attroff(COLOR_PAIR(cellInt));
			return 1;
		} else if (cellInt < 0) {
			attron(COLOR_PAIR(10 - cellInt));
			mvaddch(y, x, 'a' - cellInt);
			attroff(COLOR_PAIR(10 - cellInt));
			return 0;
		} else {
			mvaddch(y, x, ' ');
			return 0;
		}
	}

	int (*printchar_f)(float cell, int y, int x)
		= printchar__char;

	/**
	 * Print h_Output to the screen with curses
	 * Returns: Number of live cells printed
	 **/
	int print_hOutput()
	{
		int ret = 0;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				ret += printchar_f(
					MDGET(h_Output, y, x),
					y, x);
			}
		}
		return ret;
	}

	void ncurses_setup() 
	{
		initscr();
		refresh();
		if (!blocking) {
			// Don't block on user input
			nodelay(stdscr, TRUE);
		}
		if (useColor) {
			start_color();

			init_pair(1, 	COLOR_BLACK,	COLOR_GREEN);
			init_pair(2, 	COLOR_RED, 	COLOR_GREEN);
			init_pair(3, 	COLOR_YELLOW,	COLOR_GREEN);
			init_pair(4, 	COLOR_BLUE,	COLOR_GREEN);
			init_pair(5, 	COLOR_MAGENTA,	COLOR_GREEN);
			init_pair(6, 	COLOR_CYAN,	COLOR_GREEN);
			init_pair(7, 	COLOR_WHITE,	COLOR_GREEN);
			init_pair(8,	COLOR_GREEN,	COLOR_WHITE);
			init_pair(9,	COLOR_YELLOW,	COLOR_WHITE);

			init_pair(10,	COLOR_BLACK,	COLOR_RED);
			init_pair(11,	COLOR_GREEN,	COLOR_RED);
			init_pair(12,	COLOR_YELLOW,	COLOR_RED);
			init_pair(13,	COLOR_BLUE,	COLOR_RED);
			init_pair(14,	COLOR_MAGENTA,	COLOR_RED);
			init_pair(15,	COLOR_CYAN,	COLOR_RED);
			init_pair(16,	COLOR_WHITE,	COLOR_RED);
			init_pair(17,	COLOR_RED,	COLOR_WHITE);
			init_pair(18,	COLOR_YELLOW,	COLOR_WHITE);
		}
	}

	/**
	 * Attempt to reset terminal
	 **/
	void ncurses_cleanup()
	{
		if (int e = endwin()) {
			printf("endwin error %i\n", e);
		} else {
			// ANSI escape sequence to clear screen
			//  and reset cursor
			printf("\033[H\033[J");
		}
	}

	std::chrono::time_point<std::chrono::system_clock> frameClock;

	void startClock()
	{
		frameClock = std::chrono::high_resolution_clock::now();
	}

	void waitForNextFrame()
	{
		auto now = std::chrono::high_resolution_clock::now();
		auto duration = now - frameClock;
		frameClock = now;
		auto microseconds = 
			std::chrono::duration_cast
			<std::chrono::microseconds>
			(duration).count();
		if (microseconds < usPerFrame) {
			std::this_thread::sleep_for(
				std::chrono::microseconds(
					usPerFrame - microseconds));
		}
	}
}
