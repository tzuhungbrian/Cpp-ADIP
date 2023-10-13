#define _CRT_SECURE_NO_DEPRECATE
#define _USE_MATH_DEFINES

#include <stdio.h>
#include <math.h>
#include <numbers>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>

#include "Image.h"
#include "Utils.h"

using namespace std;

//---------- Private Members ----------//
complex<double> Image::dft_weight(int u, int x, int M, int v, int y, int N)
{
	complex<double> value1{ cos(2 * M_PI * u * x / M), -sin(2 * M_PI * u * x / M) };
	complex<double> value2{ cos(2 * M_PI * v * y / N), -sin(2 * M_PI * v * y / N) };
	complex<double> shiftFactor{ pow(-1, x + y), 0.0 }; // Shifting
	return shiftFactor * value1 * value2;
}

complex<double> Image::idft_weight(int u, int x, int M, int v, int y, int N)
{
	complex<double> value1{ cos(2 * M_PI * u * x / M), sin(2 * M_PI * u * x / M) };
	complex<double> value2{ cos(2 * M_PI * v * y / N), sin(2 * M_PI * v * y / N) };
	complex<double> shiftFactor{ pow(-1, x + y), 0.0 }; // Shifting
	return value1 * value2;
}

vector<int> Image::normalizeVec(vector<int> input)
{
	cout << "Normalizing..." << endl;
	vector<int> output(input.size(), 0);

	int xMax = *max_element(input.begin(), input.end());
	int xMin = *min_element(input.begin(), input.end());

	for (auto i = 0; i < input.size(); i++) {
		int normalizedVal = round(255 * (input[i] - xMin) / (xMax - xMin));
		output[i] = normalizedVal;
	}

	return output;
}

unsigned char* Image::normalizeArr(unsigned char* input)
{
	unsigned char* output = new unsigned char[size];
	for (auto i = 0; i < size; i++) {
		int normalizedVal = round(255 * (input[i] - 0) / 255);
		output[i] = normalizedVal;
	}
	return output;
}

int Image::distance(vector<int> origin, vector<int> point)
{
	int dist = sqrt(pow((point[0] - origin[0]), 2) + pow((point[1] - origin[1]), 2));
	return dist;
}

unsigned char* Image::idealFilterMask(int d0, bool isHighPass)
{
	unsigned char* outputImg = new unsigned char[size];
	vector<int> origin{ height / 2, width / 2 };
	vector<int> point(2, 0);

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			point[0] = row;
			point[1] = col;
			int dist = distance(origin, point);
			int idx = row * width + col;
			if (dist <= d0) { outputImg[idx] = 255; }
			else { outputImg[idx] = 0; }
		}
	}

	if (isHighPass == true) {
		for (auto i = 0; i < size; i++) {
			if (outputImg[i] == 255) { outputImg[i] = 0; }
			else { outputImg[i] = 255; }
		}
	}

	return outputImg;
}

uchar* Image::gaussianFilterMask(int d0, bool isHighPass)
{
	uchar* outputImg = new unsigned char[size];
	vector<int> origin{ height / 2, width / 2 };
	vector<int> point(2, 0);

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			point[0] = row;
			point[1] = col;
			int dist = distance(origin, point);
			int idx = row * width + col;
			
			float numerator = -pow(dist, 2);
			float denominator = 2 * pow(d0, 2);
			float value = 255 * exp(numerator / denominator);

			outputImg[idx] = round(value);
		}
	}

	if (isHighPass == true) {
		for (auto i = 0; i < size; i++) {
			outputImg[i] = -outputImg[i] + 255;
		}
	}

	return outputImg;
}

uchar* Image::notchMask(int d0, vector<int> p1, vector<int> p2)
{
	uchar* mask = new uchar[size];
	vector<int> point(2, 0);

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			point[0] = row;
			point[1] = col;
			int dist1 = distance(p1, point);
			int dist2 = distance(p2, point);
			int idx = row * width + col;

			if (dist1 < d0 || dist2 < d0) { mask[idx] == 0; }
			else { mask[idx] = 255; }
		}
	}
	return mask;
}

uchar* Image::bandRMask(int outerR, int innerR)
{
	uchar* mask = new uchar[size];
	vector<int> origin{ width / 2, height / 2 };
	vector<int> point(2, 0);

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			point[0] = row;
			point[1] = col;
			int dist = distance(origin, point);
			int idx = row * width + col;

			if (innerR < dist && dist <= outerR) { mask[idx] = 0; }
			else { mask[idx] = 255; }
		}
	}
	return mask;
}

vector<float> Image::homomorphicMask(float rH, float rL, int d0, float c)
{
	vector<int> origin{ width / 2, height / 2 };
	vector<int> point(2, 0);
	vector<float> output(size, 0);
	uchar* test = new uchar[size];

	float diffTerm = rH - rL;
	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			point[0] = row;
			point[1] = col;
			int dist = distance(origin, point);
			int idx = row * width + col;

			float numerator = dist * dist;
			float denominator = d0 * d0;
			float expTerm =  exp(-c * (numerator / denominator));
			float value = (diffTerm *(1 - expTerm) + rL);
			//if (value >= 1) { value = 1; }
			//else if (value <= 0) { value = 0; }

			output[idx] = value;
			//test[idx] = value * 255;
		}
	}

	//showImgCV(test, height, width, "Mask");

	return output;
}

cv::Mat Image::dftAndShift(uchar* input)
{
	cv::Mat I(height, width, CV_8UC1, arr);
	I = cv::Mat_<float>(I);

	normalize(I, I, 0, 1, cv::NORM_MINMAX);

	//expand input image to optimal size
	int m = cv::getOptimalDFTSize(I.rows);
	int n = cv::getOptimalDFTSize(I.cols);

	// on the border add zero values
	cv::Mat padded;
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
	cv::Mat spectrum;
	merge(planes, 2, spectrum);         // Add to the expanded another plane with zeros

	cv::dft(spectrum, spectrum);

	// Shifting
	cv::Mat complexPlane[2];
	cv::split(spectrum, complexPlane);

	for (auto i = 0; i < 2; i++) {
		int cx = complexPlane[i].cols / 2;
		int cy = complexPlane[i].rows / 2;

		cv::Mat q0(complexPlane[i], cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
		cv::Mat q1(complexPlane[i], cv::Rect(cx, 0, cx, cy));  // Top-Right
		cv::Mat q2(complexPlane[i], cv::Rect(0, cy, cx, cy));  // Bottom-Left
		cv::Mat q3(complexPlane[i], cv::Rect(cx, cy, cx, cy)); // Bottom-Right
		cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)

		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);
		q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
		q2.copyTo(q1);
		tmp.copyTo(q2);
	}
	merge(complexPlane, 2, spectrum);         // Add to the expanded another plane with zeros

	return spectrum;
}

cv::Mat Image::inverseDFTandShift(cv::Mat input)
{
	int cx = input.cols / 2;
	int cy = input.rows / 2;

	cv::Mat q0(input, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	cv::Mat q1(input, cv::Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(input, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(input, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
	cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	cv::Mat inversedSpectrum;
	cv::idft(input, inversedSpectrum, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

	normalize(inversedSpectrum, inversedSpectrum, 0, 1, cv::NORM_MINMAX);

	for (auto i = 0; i < height; i++) {
		for (auto j = 0; j < width; j++) {
			//cout << inversedSpectrum.at<float>(j, i) << endl;
		}
	}

	return inversedSpectrum;
}

cv::Mat Image::logDFTAndShift()

{
	cv::Mat I(height, width, CV_8UC1, arr);
	I = cv::Mat_<float>(I);

	normalize(I, I, 0, 1, cv::NORM_MINMAX);
	I += cv::Scalar::all(0.1);                    // switch to logarithmic scale
	log(I, I);

	//expand input image to optimal size
	int m = cv::getOptimalDFTSize(I.rows);
	int n = cv::getOptimalDFTSize(I.cols);

	// on the border add zero values
	cv::Mat padded;
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
	cv::Mat spectrum;
	merge(planes, 2, spectrum);         // Add to the expanded another plane with zeros

	cv::dft(spectrum, spectrum);

	// Shifting
	cv::Mat complexPlane[2];
	cv::split(spectrum, complexPlane);

	for (auto i = 0; i < 2; i++) {
		int cx = complexPlane[i].cols / 2;
		int cy = complexPlane[i].rows / 2;

		cv::Mat q0(complexPlane[i], cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
		cv::Mat q1(complexPlane[i], cv::Rect(cx, 0, cx, cy));  // Top-Right
		cv::Mat q2(complexPlane[i], cv::Rect(0, cy, cx, cy));  // Bottom-Left
		cv::Mat q3(complexPlane[i], cv::Rect(cx, cy, cx, cy)); // Bottom-Right
		cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)

		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);
		q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
		q2.copyTo(q1);
		tmp.copyTo(q2);
	}
	merge(complexPlane, 2, spectrum);         // Add to the expanded another plane with zeros

	return spectrum;
}

cv::Mat Image::shiftIDFTandExp(cv::Mat input)
{
	int cx = input.cols / 2;
	int cy = input.rows / 2;

	cv::Mat q0(input, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	cv::Mat q1(input, cv::Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(input, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(input, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
	cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	cv::Mat inversedSpectrum;
	cv::idft(input, inversedSpectrum, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

	/*normalize(inversedSpectrum, inversedSpectrum, 0, log(256), cv::NORM_MINMAX);*/
	cv::exp(inversedSpectrum, inversedSpectrum);
	//inversedSpectrum += cv::Scalar::all(-1); // Overflow to 1.x 2.x ...
	for (auto j = 0; j < height; j++) {
		for (auto i = 0; i < width; i++) {
			//cout << inversedSpectrum.at<float>(j, i) << endl;
		}
	}
	normalize(inversedSpectrum, inversedSpectrum, 0, 1, cv::NORM_MINMAX);

	return inversedSpectrum;
}

float Image::degradation(int u, int v, float k, float power)
{
	float head = -k * pow((u * u + v * v), power);
	return exp(head);
}

//---------- Public Members ----------//
Image::Image(const char* filename, int height, int width) : height(height), width(width)
{
	size = width * height;

	arr = new unsigned char[size];
	testing = 0;

	vector<complex<double>> temp(size, 0);
	freqSpect = temp;

	read(filename);
}

Image::~Image()
{
	delete[] arr;
}

void Image::read(const char* filename)
{
	FILE* imgFile = fopen(filename, "rb");
	fread(arr, sizeof(unsigned char), size, imgFile);
}

void Image::show(const char* figName)
{
	cv::Mat matrix(height, width, CV_8UC1, arr);
	cv::imshow(figName, matrix);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

void Image::save(const char* saveName)
{
	char resultRawFileName[100];         // array to hold the result.2
	strcpy(resultRawFileName, saveName); // copy string one into the result.
	strcat(resultRawFileName, ".raw");   //append string two to the result.
	FILE* saveFile = fopen(resultRawFileName, "wb");
	fwrite(arr, 1, size, saveFile);
	fclose(saveFile);

	char resultPngFileName[100];         // array to hold the result.
	strcpy(resultPngFileName, saveName); // copy string one into the result.
	strcat(resultPngFileName, ".png");   //append string two to the result.
	cv::Mat matImg(height, width, CV_8UC1, arr);
	cv::imwrite(resultPngFileName, matImg);

	std::cout << "Image has been saved Successfully as both .raw and .png!" << std::endl;
}

void Image::test()
{
	cv::Mat test = homomorphicFiltering(1.25, 0.51, 50, 0.25);
	test.convertTo(test, CV_8UC1, 255.0);
	cv::imshow("myBoy", test);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

cv::Mat Image::idealFiltering(int d0, const char* fileName, bool isHighPass, bool filteredSpectrum)
{
	unsigned char* mask = idealFilterMask(d0, isHighPass);
	cv::Mat spectrum = dftAndShift(arr); // 32FC2

	// Filtering
	for (auto row = 0; row < spectrum.rows; row++) {
		for (auto col = 0; col < spectrum.cols; col++) {
			int idx = row * width + col;

			if (mask[idx] == 0) { 
				//spectrum.at<cv::Vec2f>(col, row).val[0] = 0; // These works too.
				//spectrum.at<cv::Vec2f>(col, row).val[1] = 0;
				spectrum.at<complex<float>>(col, row) = 0;
			}
		}
	}

	if (filteredSpectrum == true) {
		uchar* magnitude = new uchar[size];
		for (auto j = 0; j < height; j++) {
			for (auto i = 0; i < width; i++) {
				int idx = i * width + j;
				magnitude[idx] = 10 * log(1 + abs(spectrum.at<complex<float>>(i, j)));
			}
		}
		saveRawAndPng(fileName, height, width, magnitude);
		showImgCV(magnitude, height, width, "Magnitude of Spectrum");
	}
	cv::Mat inverse = inverseDFTandShift(spectrum);

	return inverse;
}

cv::Mat Image::gaussianFiltering(int d0, const char* fileName, bool isHighPass, bool filteredSpectrum)
{
	unsigned char* mask = gaussianFilterMask(d0, isHighPass);
	cv::Mat spectrum = dftAndShift(arr);

	// Filtering
	for (auto row = 0; row < spectrum.rows; row++) {
		for (auto col = 0; col < spectrum.cols; col++) {
			int idx = row * width + col;
			float maskVal = (float)mask[idx] / 255;
			cv::Vec2f specVal = spectrum.at<cv::Vec2f>(col, row);

			spectrum.at<cv::Vec2f>(col, row).val[0] = round(specVal.val[0] * maskVal);
			spectrum.at<cv::Vec2f>(col, row).val[1] = round(specVal.val[1] * maskVal);
			//cout << row << ", " << col << " = "<<maskVal << endl;
		}
	}

	if (filteredSpectrum == true) {
		uchar* magnitude = new uchar[size];
		for (auto j = 0; j < height; j++) {
			for (auto i = 0; i < width; i++) {
				int idx = i * width + j;
				magnitude[idx] = 10 * log(1 + abs(spectrum.at<complex<float>>(i, j)));
			}
		}
		saveRawAndPng(fileName, height, width, magnitude);
		showImgCV(magnitude, height, width, "Magnitude of Spectrum");
	}
	cv::Mat inverse = inverseDFTandShift(spectrum);

	return inverse;
}

unsigned char* Image::dft2d()
{
	unsigned char* outputImg = new unsigned char[size];

	double value = 0.0;
	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int idx = row * width + col;
			complex<double> temp{ 0.0, 0.0 };
			for (auto x = 0; x < height; x++) {
				for (auto y = 0; y < width; y++) {
					int dftIdx = x * width + y;
					complex<double> curPix{ double(arr[dftIdx]), 0.0 };
					temp += curPix * dft_weight(row, x, height, col, y, width);
				}
			}
			freqSpect[idx] = (temp); 
			int magnitude = 20 * log(1 + abs(temp));
			outputImg[idx] = magnitude;
		}
		cout << (double)row / height << endl;
	}

	return outputImg;
}

cv::Mat Image::dft2dCV(uchar* input, bool idft = false, bool show = true)
{
	cv::Mat I(height, width, CV_8UC1, input);
                          
	//expand input image to optimal size
	int m = cv::getOptimalDFTSize(I.rows);
	int n = cv::getOptimalDFTSize(I.cols);

	// on the border add zero values
	cv::Mat padded;
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
	cv::Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	dft(complexI, complexI);            // this way the result may fit in the source matrix

	// complexI = 32FC2
	cv::Mat ifft;
	cv::idft(complexI, ifft, cv::DFT_REAL_OUTPUT);
	normalize(ifft, ifft, 0, 1, cv::NORM_MINMAX);

	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	cv::split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	cv::magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	cv::Mat magI = planes[0];
	magI += cv::Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);
	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;
	cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
	cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
	normalize(magI, magI, 0, 1, cv::NORM_MINMAX); // Transform the matrix with float values into a
											// viewable image form (float between values 0 and 1).

	if (show == true) {
		cv::imshow("Input Image", I);    // Show the result
		cv::imshow("spectrum magnitude", magI);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}

	if (idft == true) { return ifft; }
	else { return magI; }
}

unsigned char* Image::idft2d()
{
	if (freqSpect.empty() == true) {
		cout << "No Frequency Spectrum!" << endl;
		return 0;
	}

	unsigned char* outputImg = new unsigned char[size];
	vector<int> outputVec(size, 0);

	double value = 0.0;
	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int idx = row * width + col;
			complex<double> temp{ 0.0, 0.0 };
			for (auto u = 0; u < height; u++) {
				for (auto v = 0; v < width; v++) {
					int idftIdx = u * width + v;
					complex<double> curPix = freqSpect[idftIdx];
					temp += curPix * idft_weight(row, u, height, col, v, width);
				}
			}
			int magnitude = log(1 + abs(temp));
			outputVec[idx] = magnitude;
		}
		cout << (double)row / height << endl;
	}

	outputVec = normalizeVec(outputVec);
	for (auto i = 0; i < size; i++) { outputImg[i] = outputVec[i]; }

	return outputImg;
}

cv::Mat Image::applyWatermark(uchar* watermark, float k, const char* fileName, bool filteredSpectrum)
{
	cv::Mat spectrum = dftAndShift(arr);
	cv::Mat showOnly;
	spectrum.copyTo(showOnly);

	// Applying watermark
	for (auto row = 0; row < spectrum.rows; row++) {
		for (auto col = 0; col < spectrum.cols; col++) {
			int idx = row * width + col;
			float watermarkVal = (float)watermark[idx] * k;
			cv::Vec2f specVal = spectrum.at<cv::Vec2f>(col, row);

			spectrum.at<cv::Vec2f>(col, row).val[0] = round(specVal.val[0] + watermarkVal);
			spectrum.at<cv::Vec2f>(col, row).val[1] = round(specVal.val[1] + watermarkVal);
		}
	}

	if (filteredSpectrum == true) {
		uchar* magnitude = new uchar[size];
		for (auto j = 0; j < height; j++) {
			for (auto i = 0; i < width; i++) {
				int idx = i * width + j;
				float watermarkVal = (float)watermark[idx] * k;
				magnitude[idx] = 10 * log(1 + abs(showOnly.at<complex<float>>(i, j) + watermarkVal));
			}
		}
		saveRawAndPng(fileName, height, width, magnitude);
		showImgCV(magnitude, height, width, "Magnitude of Spectrum");
	}
	cv::Mat inverse = inverseDFTandShift(spectrum);

	return inverse;
}

cv::Mat Image::homomorphicFiltering(float rH, float rL, int d0, float c)
{
	// 1. Log, DFT and shift
	cv::Mat spectrum = logDFTAndShift();
	
	// 2. Apply homomorphic
	vector<float> mask = homomorphicMask(rH, rL, d0, c);
	for (auto row = 0; row < spectrum.rows; row++) {
		for (auto col = 0; col < spectrum.cols; col++) {
			int idx = row * width + col;
			float maskVal = (float)mask[idx];
			cv::Vec2f specVal = spectrum.at<cv::Vec2f>(col, row);

			spectrum.at<cv::Vec2f>(col, row).val[0] = round(specVal.val[0] * maskVal);
			spectrum.at<cv::Vec2f>(col, row).val[1] = round(specVal.val[1] * maskVal);
			
			//cout << "Coordinate = " << col << ", " << row << endl;
			//cout << specVal << " x " << maskVal << " = " << specVal * maskVal << endl;
		}
	}
	
	uchar* magnitude = new uchar[size];
	for (auto j = 0; j < height; j++) {
		for (auto i = 0; i < width; i++) {
			int idx = i * width + j;
			magnitude[idx] = 10 * log(1 + abs(spectrum.at<complex<float>>(i, j)));
		}
	}
	//showImgCV(magnitude, height, width, "Magnitude of Spectrum");

	
	// 3. Exp, Inverse shift and IDFT
	cv::Mat inverse = shiftIDFTandExp(spectrum);

	return inverse;
}

cv::Mat Image::notchFiltering(int d0, vector<int> p1, vector<int> p2)
{
	unsigned char* mask = notchMask(d0, p1, p2);
	cv::Mat spectrum = dftAndShift(arr);

	// Filtering
	for (auto row = 0; row < spectrum.rows; row++) {
		for (auto col = 0; col < spectrum.cols; col++) {
			int idx = row * width + col;
			float maskVal = (float)mask[idx] / 255;
			cv::Vec2f specVal = spectrum.at<cv::Vec2f>(col, row);

			spectrum.at<cv::Vec2f>(col, row).val[0] = round(specVal.val[0] * maskVal);
			spectrum.at<cv::Vec2f>(col, row).val[1] = round(specVal.val[1] * maskVal);
		}
	}

	//uchar* magnitude = new uchar[size];
	//for (auto j = 0; j < height; j++) {
	//	for (auto i = 0; i < width; i++) {
	//		int idx = i * width + j;
	//		magnitude[idx] = 10 * log(1 + abs(spectrum.at<complex<float>>(i, j)));
	//	}
	//}
	//showImgCV(magnitude, height, width, "Magnitude of Spectrum");
	
	cv::Mat inverse = inverseDFTandShift(spectrum);

	return inverse;
}

cv::Mat Image::bandReject(int outerR, int innerR)
{
	unsigned char* mask = bandRMask(outerR, innerR);
	cv::Mat spectrum = dftAndShift(arr);

	// Filtering
	for (auto row = 0; row < spectrum.rows; row++) {
		for (auto col = 0; col < spectrum.cols; col++) {
			int idx = row * width + col;
			float maskVal = (float)mask[idx] / 255;
			cv::Vec2f specVal = spectrum.at<cv::Vec2f>(col, row);

			spectrum.at<cv::Vec2f>(col, row).val[0] = round(specVal.val[0] * maskVal);
			spectrum.at<cv::Vec2f>(col, row).val[1] = round(specVal.val[1] * maskVal);
		}
	}

	uchar* magnitude = new uchar[size];
	for (auto j = 0; j < height; j++) {
		for (auto i = 0; i < width; i++) {
			int idx = i * width + j;
			magnitude[idx] = 10 * log(1 + abs(spectrum.at<complex<float>>(i, j)));
		}
	}
	showImgCV(magnitude, height, width, "Magnitude of Spectrum");

	cv::Mat inverse = inverseDFTandShift(spectrum);

	return inverse;
}

cv::Mat Image::inverseFiltering(int r, float threshold)
{
	// DFT -> Shifting -> F_hat = G / H + N / H -> Inverse Shifting -> IDFT
	cv::Mat spectrum = dftAndShift(arr);
	uchar* mask = idealFilterMask(r, false);
	//showImgCV(mask, height, width, "Mask");
	vector<double> deg(size, 0);
	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int idx = row * width + col;

			int maskVal = (double)mask[idx];
			cv::Vec2f specVal = spectrum.at<cv::Vec2f>(col, row);
			auto degradeVal = degradation(row - (height / 2), col - (width / 2), 0.003, 0.8333);

			if (maskVal != 0) {
				spectrum.at<cv::Vec2f>(col, row).val[0] = round(specVal.val[0] / degradeVal);
				spectrum.at<cv::Vec2f>(col, row).val[1] = round(specVal.val[1] / degradeVal);
			}
		}
	}
	cv::Mat inverse = inverseDFTandShift(spectrum);

	return inverse;
}

cv::Mat Image::wienerFiltering(double constK)
{
	// DFT -> Shifting -> F_hat = G / H + N / H -> Inverse Shifting -> IDFT
	cv::Mat spectrum = dftAndShift(arr);
	//showImgCV(mask, height, width, "Mask");
	vector<double> deg(size, 0);
	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int idx = row * width + col;

			cv::Vec2f specVal = spectrum.at<cv::Vec2f>(col, row);
			auto Huv = degradation(row - (height / 2), col - (width / 2), 0.003, 0.8333);
			auto conjHuv = conj(Huv);
			auto absHuv = abs(Huv) * abs(Huv);

			auto term = (absHuv / (absHuv + constK)) / Huv;

			spectrum.at<cv::Vec2f>(col, row).val[0] = round(specVal.val[0] * term);
			spectrum.at<cv::Vec2f>(col, row).val[1] = round(specVal.val[1] * term);
		}
	}
	cv::Mat inverse = inverseDFTandShift(spectrum);

	return inverse;
}

