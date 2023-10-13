#define _CRT_SECURE_NO_DEPRECATE
#define _USE_MATH_DEFINES

#include <string>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>

#include "Utils.h"

using namespace std;

void popOutMenu()
{
	cout
		<< "0: Exit the program." << endl << endl

		<< "Question 1: Homomorphic Filtering (30%)" << endl
		<< "-----------------------------------" << endl
		<< "1: 1 - Homomorphic Filtering" << endl << endl

		<< "Question 2: Periodic Noise (30%)" << endl
		<< "-----------------------------------" << endl
		<< "2: 2_1 - Denoise Apple by Notch Filtering" << endl
		<< "3: 2_2 - Denoise Apple by Band-Reject Filtering" << endl << endl

		<< "Question 3: Deblurring (40%)" << endl
		<< "-----------------------------------" << endl
		<< "4: 3 - Deblurring" << endl << endl

		<< endl;
}

void saveRawAndPng(const char* saveName, int height, int width, unsigned char* saveImgArr)
/** @brief Save image as both raw and png files.
 *
 */
{
	int size = width * height;

	char resultRawFileName[100];         // array to hold the result.
	strcpy(resultRawFileName, saveName); // copy string one into the result.
	strcat(resultRawFileName, ".raw");   //append string two to the result.
	FILE* saveFile = fopen(resultRawFileName, "wb");
	fwrite(saveImgArr, 1, size, saveFile);
	fclose(saveFile);

	char resultPngFileName[100];         // array to hold the result.
	strcpy(resultPngFileName, saveName); // copy string one into the result.
	strcat(resultPngFileName, ".png");   //append string two to the result.
	cv::Mat matImg(height, width, CV_8UC1, saveImgArr);
	cv::imwrite(resultPngFileName, matImg);

	std::cout << "Image has been saved Successfully as both .raw and .png!" << std::endl;
}

void showImgCV(unsigned char* imgArr, int height, int width, std::string figName)
{
	cv::Mat matrix(height, width, CV_8UC1, imgArr);
	cv::imshow(figName, matrix);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

string typeOfMat(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

float mse(unsigned char* aImg, unsigned char* bImg, int height, int width, bool print)
{
	int size = height * width;
	float pixelScore = 0;

	for (auto row = 0; row != height; ++row) {
		for (auto col = 0; col != width; ++col) {
			int index = row * width + col;
			pixelScore += pow((aImg[index] - bImg[index]), 2);
		}
	}
	float mseScore = pixelScore / size;
	if (print == true) { cout << "MSE Score is: " << mseScore << endl; }

	return mseScore;
}

vector<vector<float>> genGaussianKernel(int maskSize, float sigma, float mean)
{
	// Initialize 2-D Vector
	vector<vector<float>> mask(maskSize);
	for (auto i = 0; i < maskSize; i++) { mask[i].resize(maskSize); }

	double s = 2.0 * sigma * sigma;
	double sum = 0.0; // for normalization

	int d = floor((float)maskSize / 2);
	// generating kernel
	for (int x = -d; x <= d; x++) {
		for (int y = -d; y <= d; y++) {
			double r = sqrt(x * x + y * y);
			mask[x + d][y + d] = (exp(-(r * r) / s)) / (M_PI * s);
			sum += mask[x + d][y + d];
		}
	}

	// normalising the Kernel
	for (int i = 0; i < maskSize; ++i)
		for (int j = 0; j < maskSize; ++j)
			mask[i][j] /= sum;

	return mask;
}
unsigned char* gaussianBlur(unsigned char* inputImg, int height, int width, int gaussianSize, float sigma, float mean)
{
	int size = height * width;
	unsigned char* outputImg = new unsigned char[size];

	vector<vector<float>> mask = genGaussianKernel(gaussianSize, sigma, mean);


	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {

			int d = floor((float)gaussianSize / 2);
			float newPixel = 0.0;
			for (auto winRow = 0; winRow < gaussianSize; winRow++) {
				int sampleRow = row - d + winRow;
				if (sampleRow >= height - 1) { sampleRow = height - 1; } // Bottom Boundary Case

				for (auto winCol = 0; winCol < gaussianSize; winCol++) {
					int sampleCol = col - d + winCol;
					if (sampleCol >= width - 1) { sampleCol = width - 1; } // Right Boundary Case

					int sampleIdx = sampleRow * width + sampleCol;
					if (sampleIdx <= 0) { sampleIdx = 0; } // Prevent from out-of-index error

					newPixel += mask[winRow][winCol] * inputImg[sampleIdx];
				}
			}
			int idx = row * width + col;
			outputImg[idx] = newPixel;
		}
	}

	return outputImg;
}

cv::Mat checkSpectrum(uchar* input, int height, int width, bool show)
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

	return magI;
}
