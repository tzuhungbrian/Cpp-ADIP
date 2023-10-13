#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <vector>

using namespace std;

class Image
{
private:
	complex<double> dft_weight(int u, int x, int M, int v, int y, int N);
	complex<double> idft_weight(int u, int x, int M, int v, int y, int N);
	vector<int> normalizeVec(vector<int> input);
	unsigned char* normalizeArr(unsigned char* input);
	int distance(vector<int> origin, vector<int> point);
	unsigned char* idealFilterMask(int d0, bool isHighPass);
	uchar* gaussianFilterMask(int d0, bool isHighPass);
	cv::Mat dftAndShift(uchar* input);
	cv::Mat inverseDFTandShift(cv::Mat input);

public:
	int height;
	int width;
	int size;
	unsigned char* arr; // Store original data
	vector<complex<double>> freqSpect; // Store frequency spectrum image
	int testing;

	Image(const char* filename, int height, int width);
	~Image();

	void read(const char* filename);
	void show(const char* figName);
	void save(const char* saveName);
	void test();

	unsigned char* dft2d();
	cv::Mat dft2dCV(uchar* input, bool idft, bool show);
	unsigned char* idft2d();

	cv::Mat idealFiltering(int d0, const char* fileName,
		bool isHighPass = false, bool filteredSpectrum = false);
	cv::Mat gaussianFiltering(int d0, const char* fileName,
		bool isHighPass = false, bool filteredSpectrum = false);

	cv::Mat applyWatermark(uchar* watermark, float k, const char* fileName, bool filteredSpectrum);
};