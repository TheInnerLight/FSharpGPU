const int MAX_BLOCKS = 65535;
const int MAX_THREADS = 512;

struct ThreadBlocks{
	int threadCount;
	int blockCount;
	int thrBlockCount;
	int loopCount;
	int N;
};