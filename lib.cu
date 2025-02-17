#include <iostream>
#include <stdio.h>
#include <fstream>

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

#define WIDTH 80
#define HEIGHT 24

// Get element from array of size WIDTH*HEIGHT 
//  treated as 2-dimensional
#define MDGET(A,Y,X) A[(Y)*(WIDTH)+(X)]

extern float *d_Input;
extern float *d_Output;
extern float *h_Input;
extern float *h_Output;

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
/**
 * Wrappers for cuda calls to check for errors
 **/
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define gpuErrChkF(ans, f) { \
	gpuAssert((ans), __FILE__, __LINE__, true, (f)); \
}

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


/**
 * Get an arbitrary ASCII char from 33 to 126,
 *  based on deterministic modulo of input
 **/
char fancy_char_yx(int y, int x) 
{
	// return '#';
	
	static int fancyMax = 126;
	static int fancyMin = 33;
	int n = y*WIDTH + x;
	n %= (fancyMax - fancyMin);
	return n + fancyMin;
}

/**
 * Print h_Output to the screen with curses
 * Returns: Number of live cells printed
 **/
int print_hOutput()
{
	int ret = 0;
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			float output = MDGET(h_Output, y, x);
			char ch = ' ';
			if (output) {
				ret++;
				ch = fancy_char_yx(y, x);
			}
			// ncurses takes care of 
			//  efficient screen updating
			mvaddch(y, x, ch);
		}
	}
	return ret;
}

void ncurses_setup() 
{
	initscr();
	refresh();
#if !BLOCKING
	// Don't block on user input
	nodelay(stdscr, TRUE);
#endif
}

/**
 * Attempt to reset terminal
 **/
void ncurses_cleanup()
{
	if (int e = endwin()) {
		printf("endwin error %i\n", e);
	}
}

