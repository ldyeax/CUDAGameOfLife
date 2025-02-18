// Get element from array of size width*height
//  treated as 2-dimensional
#define MDGET(A,Y,X) A[(Y)*(width)+(X)]

/**
 * Wrappers for cuda calls to check for errors
 **/
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define gpuErrChkF(ans, f) { \
        gpuAssert((ans), __FILE__, __LINE__, true, (f)); \
}


namespace CUDAGameOfLife
{
	extern float *d_Input;
	extern float *d_Output;
	extern float *h_Input;
	extern float *h_Output;

	extern int width;
	extern int height;
	
	/**
	 * Blocking (wait for keypress to continue)
	 * or non-blocking (keep simulating unattended)	
	 **/
	extern bool blocking;
	/**
          * If blocking, wait this long before showing next render
          **/
	extern int usPerFrame;

	extern bool useColori;

	/**
	 * Size of all our input/output arrays
	 **/
	extern size_t mdsize;

	/**
	 * Get width/height and init mdsize
	 **/
	void initWidthHeight();

	/**
	 * If str is numeric, set result to the integer value;
	 *  otherwise return false and do nothing
	 **/
	bool testNumeric(char* str, int* result);

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
                bool abort,
                void (*beforePrint)());

	/**
	 * Read plaintext .cells format file into h_Input
	 * https://conwaylife.com/wiki/Plaintext
	 **/
	extern int readCellsFile(char* filename, int offsetY, int offsetX);

	void printHelp();

	int parseArguments();

	extern int (*printchar_f)(float cell, int y, int x);
	extern char printchar__char__char;
	int printchar__char(float, int, int);
	int printchar__fancy(float, int, int);
	int printchar__neighbors(float, int, int);
	int printchar__color(float, int, int);
	
	int print_hOutput();

	void ncurses_setup();
	void ncurses_cleanup();

	void startClock();
	void waitForNextFrame();
}
