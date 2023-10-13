#define _CRT_SECURE_NO_DEPRECATE
#define _USE_MATH_DEFINES

#include <stdio.h>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <math.h>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

//---------- Miscellaneous ----------//
void popOutMenu()
{
	cout
		<< "0: Exit the program." << endl << endl

		<< "Question 1: Histogram Processing (40%)" << endl
		<< "-----------------------------------" << endl
		<< "1: 1_a - Histogram Match" << endl
		<< "2: 1_b - Local Histogram Equalization" << endl << endl

		<< "Question 2: Image Smoothing & Sharpening (30%)" << endl
		<< "-----------------------------------" << endl
		<< "3: 2_a - Gaussian Blur and ROI" << endl
		<< "4: 2_b - Sharpen Image by ROI" << endl << endl

		<< "Question 3: Edge Detection (30%)" << endl
		<< "-----------------------------------" << endl
		<< "5: 3_a - Laplacian and Sobel Filtering" << endl
		<< "6: 3_b - Extract Receipt" << endl << endl

		<< endl;
}
void showImgCV(unsigned char* imgArr, int height, int width, std::string figName)
{
	cv::Mat matrix(height, width, CV_8UC1, imgArr);
	cv::imshow(figName, matrix);
	cv::waitKey(0);
	cv::destroyAllWindows();
}
unsigned char* readImg(const char* fileNameImg, int height, int width, bool debug=false)
{
	int size = height * width;
	unsigned char* imgArr = new unsigned char[size];
	FILE* imgFile = fopen(fileNameImg, "rb");
	fread(imgArr, sizeof(unsigned char), size, imgFile);
	if (debug == true) { showImgCV(imgArr, height, width, "Debug mode"); }
	return imgArr;
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
//---------- Question 1 ----------//
vector<int> computeHist(unsigned char* img, int height, int width)
{
	vector<int> hist(256, 0);

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int idx = width * row + col;
			hist[img[idx]]++;
		}
	}
	return hist;
}
vector<float> computeCDF(unsigned char* img, int height, int width)
{
	int size = height * width;
	vector<int> imgHist = computeHist(img, height, width);
	vector<float> cdfVec(256, 0);

	int cumulation = 0;
	for (auto idx = 0; idx < imgHist.size(); idx++) {
		float newPosition;
		cumulation += imgHist[idx];
		cdfVec[idx] = (float)cumulation / size;
	}
	return cdfVec;
}
unsigned char* histogramMatch(unsigned char* inputImg, int inputH, int inputW,
							  unsigned char* matchImg, int matchH, int matchW)
{
	int inputSize = inputH * inputW;
	unsigned char* outputImg = new unsigned char[inputSize];

	vector<float> inputCDF = computeCDF(inputImg, inputH, inputW);
	vector<float> matchCDF = computeCDF(matchImg, matchH, matchW);

	for (auto row = 0; row < inputH; row++) {
		for (auto col = 0; col < inputW; col++) {
			int idx = inputW * row + col;
			int curPix = inputImg[idx];

			float curProb = inputCDF[curPix];
			for (auto x = 0; x < inputCDF.size(); x++) {
				if (matchCDF[x] <= curProb < matchCDF[x + 1]) {
					outputImg[idx] = x;
					break;
				}
			}
		}
	}
	return outputImg;
}

vector<int> getTransformVec(unsigned char* img, int height, int width)
{
	int size = height * width;
	vector<int> srcHist = computeHist(img, height, width);
	vector<int> transform(256, 0);

	int cumulation = 0;
	for (auto idx = 0; idx < srcHist.size(); idx++) {
		float newPosition;
		cumulation += srcHist[idx];
		if (srcHist[idx] == 0 && newPosition == 0) { continue; }
		if (idx == 0) { newPosition = 0; }
		else { newPosition = 255 * (float)cumulation / size; }
		transform[idx] = round(newPosition);
	}
	delete[] img;
	return transform;
}
unsigned char* localHE(unsigned char* inputImg, int height, int width, int windowLen)
{
	int size = height * width;
	unsigned char* outputImg = new unsigned char[size];

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int idx = row * width + col;
			
			int windowSize = windowLen * windowLen;
			unsigned char* window = new unsigned char[windowSize];
			// Fill in the value in current window
			int winIdx = 0;
			int d = floor((float)windowLen / 2);
			for (auto winRow = 0; winRow < windowLen; winRow++) {
				int sampleRow = row - d + winRow;
				if (sampleRow >= height - 1) { sampleRow = height - 1; } // Bottom Boundary Case

				for (auto winCol = 0; winCol < windowLen; winCol++) {
					int sampleCol = col - d + winCol;
					if (sampleCol >= width - 1) { sampleCol = width - 1; } // Right Boundary Case

					int sampleIdx = sampleRow * width + sampleCol;
					if (sampleIdx <= 0) { sampleIdx = 0; } // Prevent from out-of-index error

					window[winIdx] = inputImg[sampleIdx];
					winIdx += 1;
				}
			}

			// Apply histogram equalization on this window
			vector<int> transform = getTransformVec(window, windowLen, windowLen);
			outputImg[idx] = transform[inputImg[idx]];
			
		}
	}
	return outputImg;
}
cv::Mat plotHist(vector<int> hist, float scale = 0.8, bool showFig = false)
{
	int size = 512 * 512;
	unsigned char* outputImg = new unsigned char[size];

	int maxValue = *max_element(hist.begin(), hist.end()); // Max value in hist
	int maxHeight = round(512 * scale); // The highest bar in hist

	float barScale = (float)maxHeight / maxValue; // Everybody multiplied by this will be scaled to fit in the figure
	cv::Mat3b matOutput = cv::Mat3b(512, 512, cv::Vec3b(0, 0, 0));

	// Remember that openCV coordinates from top
	// Loop over hist vector
	for (auto xAxis = 0; xAxis < hist.size(); xAxis++) {
		int barHeight = hist[xAxis] * barScale;
		cv::Point pt1(xAxis * 2, 512 - 1 - barHeight);
		cv::Point pt2(xAxis * 2 + 1, 512 - 1);
		cv::rectangle(matOutput, pt1, pt2, (xAxis % 2) ? cv::Scalar(0, 100, 255) : cv::Scalar(0, 0, 255), cv::FILLED);
	}

	if (showFig == true) {
		cv::imshow("Histogram", matOutput);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
	return matOutput;
}

//---------- Question 2 ----------//
vector<vector<float>> genGaussianKernel(int maskSize, float sigma = 1.0, float mean = 0.0)
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
unsigned char* gaussianBlur(unsigned char* inputImg, int height, int width, int gaussianSize, float sigma = 1.0, float mean = 0.0)
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
unsigned char* thersholding(unsigned char* inputImg, int height, int width, float threshold)
{
	int size = height * width;
	unsigned char* outputImg = new unsigned char[size];

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int idx = row * width + col;
			float normPix = (float)inputImg[idx] / 255;

			if (normPix >= threshold) { outputImg[idx] = 255; }
			else { outputImg[idx] = 0; }
		}
	}
	return outputImg;
}
unsigned char* sharpenROI(unsigned char* inputImg, unsigned char* roi, unsigned char* blurredImg, int height, int width)
{
	int size = height * width;
	unsigned char* outputImg = new unsigned char[size];

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int idx = row * width + col;
			if (roi[idx] == 255) { outputImg[idx] = inputImg[idx]; }
			else { outputImg[idx] = blurredImg[idx]; }
		}
	}
	return outputImg;
}

//---------- Question 3 ----------//
unsigned char* laplacian(unsigned char* inputImg, int height, int width)
{
	int size = height * width;
	unsigned char* outputImg = new unsigned char[size];

	vector<vector<float>> mask{
		{-1, -1, -1},
		{-1, 8, -1},
		{-1, -1, -1}
	};

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int newPixel = 0.0;
			for (auto winRow = 0; winRow < 3; winRow++) {
				int sampleRow = row - 1 + winRow;
				if (sampleRow >= height - 1) { sampleRow = height - 1; } // Bottom Boundary Case
				for (auto winCol = 0; winCol < 3; winCol++) {
					int sampleCol = col - 1 + winCol;
					if (sampleCol >= width - 1) { sampleCol = width - 1; } // Right Boundary Case
					else if (sampleCol < 0) { sampleCol = 0; } // Left boundary case
					int sampleIdx = sampleRow * width + sampleCol;
					if (sampleIdx <= 0) { sampleIdx = 0; } // Prevent from out-of-index error

					newPixel += mask[winRow][winCol] * inputImg[sampleIdx];
				}
			}
			if (newPixel < 0) { newPixel = 0; }
			else if (newPixel > 255) { newPixel = 255; }
			int idx = row * width + col;
			outputImg[idx] = newPixel;
		}
	}

	return outputImg;
}
unsigned char* sobel(unsigned char* inputImg, int height, int width)
{
	int size = height * width;
	unsigned char* G_x = new unsigned char[size];
	unsigned char* G_y = new unsigned char[size];
	unsigned char* outputImg = new unsigned char[size];

	vector<vector<float>> verMask{
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1}
	};
	vector<vector<float>> horMask{
		{1, 2, 1},
		{0, 0, 0},
		{-1, -2, -1}
	};


	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int d = 1;
			float newPixel = 0.0;
			for (auto winRow = 0; winRow < 3; winRow++) {
				int sampleRow = row - d + winRow;
				if (sampleRow >= height - 1) { sampleRow = height - 1; } // Bottom Boundary Case

				for (auto winCol = 0; winCol < 3; winCol++) {
					int sampleCol = col - d + winCol;
					if (sampleCol >= width - 1) { sampleCol = width - 1; } // Right Boundary Case
					else if (sampleCol < 0) { sampleCol = 0; }
					int sampleIdx = sampleRow * width + sampleCol;
					if (sampleIdx <= 0) { sampleIdx = 0; } // Prevent from out-of-index error

					newPixel += verMask[winRow][winCol] * inputImg[sampleIdx];
				}
			}
			if (newPixel < 0) { newPixel = 0; }
			else if (newPixel > 255) { newPixel = 255; }
			int idx = row * width + col;
			G_x[idx] = newPixel;
		}
	}

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {

			int d = 1;
			float newPixel = 0.0;
			for (auto winRow = 0; winRow < 3; winRow++) {
				int sampleRow = row - d + winRow;
				if (sampleRow >= height - 1) { sampleRow = height - 1; } // Bottom Boundary Case

				for (auto winCol = 0; winCol < 3; winCol++) {
					int sampleCol = col - d + winCol;
					if (sampleCol >= width - 1) { sampleCol = width - 1; } // Right Boundary Case
					else if (sampleCol < 0) { sampleCol = 0; }
					int sampleIdx = sampleRow * width + sampleCol;
					if (sampleIdx <= 0) { sampleIdx = 0; } // Prevent from out-of-index error

					newPixel += horMask[winRow][winCol] * inputImg[sampleIdx];
				}
			}
			if (newPixel < 0) { newPixel = 0; }
			else if (newPixel > 255) { newPixel = 255; }
			int idx = row * width + col;
			G_y[idx] = newPixel;
		}
	}

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int idx = row * width + col;
			float pix = sqrt(powf(G_x[idx], 2) + powf(G_y[idx], 2));
			outputImg[idx] = pix;
		}
	}

	return outputImg;
}

unsigned char* powerLaw(unsigned char* srcArr, int height, int width, float gamma, float c = 1.0)
{
	int size = height * width;
	unsigned char* desArr = new unsigned char[size];

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int index = row * width + col;
			desArr[index] = c * pow((float)srcArr[index] / 255, gamma) * 255;
		}
	}
	return desArr;
}
unsigned char* addImg(unsigned char* imgA, unsigned char* imgB, int height, int width)
{
	int size = height * width;
	unsigned char* outputImg = new unsigned char[size];

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int idx = width * row + col;
			outputImg[idx] = imgA[idx] + imgB[idx]*0.1;
		}
	}
	return outputImg;
}

int main()
{
	unsigned char* imglena512 = readImg("lena_512.raw", 512, 512);
	unsigned char* imgCat512 = readImg("cat_512.raw", 512, 512);
	unsigned char* imgCatch = readImg("catch_300x168.raw", 168, 300);
	unsigned char* imgRain = readImg("rain_769x512.raw", 512, 769);
	unsigned char* imgUmbrella = readImg("umbrella_800x512.raw", 512, 800);
	unsigned char* imgRoof = readImg("roof_512.raw", 512, 800);
	unsigned char* imgRoofNoise = readImg("roof_noise_512.raw", 512, 800);
	unsigned char* imgReceipt = readImg("receipt_512x686.raw", 686, 512);

	//---------- Homework ----------//
	popOutMenu();
	while (true)
	{
		cout << endl << "Enter the Question Number to Show Answer:" << endl;
		int select = 0;
		cin >> select;

		if (select == 0) { break; }

		// 1_a 
		else if (select == 1) {
			// 1. Show the image of cat_512.raw after applying histogram match from 
			//    catch_300x168.raw and rain_769*512.raw.
			unsigned char* catFromCatch = histogramMatch(imgCat512, 512, 512, imgCatch, 168, 300);
			unsigned char* catFromRain = histogramMatch(imgCat512, 512, 512, imgRain, 512, 769);

			cv::Mat matCFC(512, 512, CV_8UC1, catFromCatch);
			cv::Mat matCFR(512, 512, CV_8UC1, catFromRain);
			cv::imshow("Cat By Catch", matCFC);
			cv::imshow("Cat By Rain", matCFR);
			cv::waitKey(0);
			cv::destroyAllWindows();

			// 2. Compare the histogram of cat_512.raw before and after.
			vector<int> catHistVec = computeHist(imgCat512, 512, 512);
			vector<int> catchHistVec = computeHist(imgCatch, 512, 512);
			vector<int> rainHistVec = computeHist(imgRain, 512, 512);
			vector<int> cfcHistVec = computeHist(catFromCatch, 512, 512);
			vector<int> cfrHistVec = computeHist(catFromRain, 512, 512);

			cv::Mat catHist = plotHist(catHistVec, 0.8);
			cv::Mat catchHist = plotHist(catchHistVec, 0.8);
			cv::Mat rainHist = plotHist(rainHistVec, 0.8);
			cv::Mat cfcHist = plotHist(cfcHistVec, 0.8);
			cv::Mat cfrHist = plotHist(cfrHistVec, 0.8);

			cv::imshow("Original Cat", catHist);
			cv::imshow("Original Catch", catchHist);
			cv::imshow("Original Rain", rainHist);
			cv::imshow("Cat By Catch", cfcHist);
			cv::imshow("Cat By Rain", cfrHist);
			cv::waitKey(0);
			cv::destroyAllWindows();

			saveRawAndPng("1_a_cat_catch", 512, 512, catFromCatch);
			saveRawAndPng("1_a_cat_rain", 512, 512, catFromRain);
			cv::imwrite("1_a_cat_hist.png", catHist);
			cv::imwrite("1_a_catch_hist.png", catchHist);
			cv::imwrite("1_a_rain_hist.png", rainHist);
			cv::imwrite("1_a_cat_catch_hist.png", cfcHist);
			cv::imwrite("1_a_cat_rain_hist.png", cfrHist);



			delete[] catFromCatch, catFromRain;
		}

		/** 1_b
		 *  Apply local histogram equalization on rain.raw. Try 3 different masks.
		 *  Show the output of 3 images.
		 *  What area has been enhanced but not shown in the global one?
		 */
		else if (select == 2) {
			unsigned char* lheRainA = localHE(imgRain, 512, 769, 3);
			unsigned char* lheRainB = localHE(imgRain, 512, 769, 21);
			unsigned char* lheRainC = localHE(imgRain, 512, 769, 100);

			saveRawAndPng("1_b_lheRainA", 512, 769, lheRainA);
			saveRawAndPng("1_b_lheRainB", 512, 769, lheRainB);
			saveRawAndPng("1_b_lheRainC", 512, 769, lheRainC);

			cv::Mat matRain(512, 769, CV_8UC1, imgRain);
			cv::Mat matA(512, 769, CV_8UC1, lheRainA);
			cv::Mat matB(512, 769, CV_8UC1, lheRainB);
			cv::Mat matC(512, 769, CV_8UC1, lheRainC);

			cv::imshow("Original", matRain);
			cv::imshow("3 x 3", matA);
			cv::imshow("21 x 21", matB);
			cv::imshow("100 x 100", matC);
			cv::waitKey(0);
			cv::destroyAllWindows();

			vector<int> aHist = computeHist(lheRainA, 512, 769);
			cv::Mat matAHist = plotHist(aHist);
			cv::imwrite("1_b_aHist.png", matAHist);

			vector<int> bHist = computeHist(lheRainB, 512, 769);
			cv::Mat matBHist = plotHist(bHist);
			cv::imwrite("1_b_bHist.png", matBHist);

			vector<int> cHist = computeHist(lheRainC, 512, 769);
			cv::Mat matCHist = plotHist(cHist);
			cv::imwrite("1_b_cHist.png", matCHist);

		}

		/** 2
		 *  (a) Extract ROI on umbrella by
		 *     1. Low-Pass Gaussian
		 *	   2. Thresholding
		 *  (b) Sharpen image by ROI 
		 */
		else if (select == 3 || select == 4) {
			unsigned char* blurredUmb = gaussianBlur(imgUmbrella, 512, 800, 27, 30.0);
			unsigned char* roiUmb = thersholding(blurredUmb, 512, 800, 0.525);
			unsigned char* sharpenImg = sharpenROI(imgUmbrella, roiUmb, blurredUmb, 512, 800);

			saveRawAndPng("2_a_blurr", 512, 800, blurredUmb);
			saveRawAndPng("2_a_roi", 512, 800, roiUmb);
			saveRawAndPng("2_b", 512, 800, sharpenImg);

			if (select == 3) {
				cv::Mat matUmb(512, 800, CV_8UC1, imgUmbrella);
				cv::Mat matBlurred(512, 800, CV_8UC1, blurredUmb);
				cv::Mat matROI(512, 800, CV_8UC1, roiUmb);

				cv::imshow("Original", matUmb);
				cv::imshow("Gaussian £m=30, £g=0", matBlurred);
				cv::imshow("ROI with threshold=0.525", matROI);

				cv::waitKey(0);
				cv::destroyAllWindows();
			}
			if (select == 4) {
				cv::Mat matROI(512, 800, CV_8UC1, sharpenImg);
				cv::imshow("ROI", matROI);
				cv::waitKey(0);
				cv::destroyAllWindows();
			}
		}

		/** 3_a 
		 *  Apply 
		 *  (1) Lapalcian filtering and (2) Sobel filtering on roof and roof noise.
		 *  Discuss the noise sensitivity, and how to process boundary.
		 */
		else if (select == 5) {
			unsigned char* lapRoof = laplacian(imgRoof, 512, 512);
			unsigned char* lapRoofNoise = laplacian(imgRoofNoise, 512, 512);
			unsigned char* sobRoof = sobel(imgRoof, 512, 512);
			unsigned char* sobRoofNoise = sobel(imgRoofNoise, 512, 512);

			saveRawAndPng("3_a_lapRoof", 512, 512, lapRoof);
			saveRawAndPng("3_a_lapRoofNoise", 512, 512, lapRoofNoise);
			saveRawAndPng("3_a_sobRoof", 512, 512, sobRoof);
			saveRawAndPng("3_a_sobRoofNoise", 512, 512, sobRoofNoise);


			cv::Mat matRoof(512, 512, CV_8UC1, imgRoof);
			cv::Mat matLapRoof(512, 512, CV_8UC1, lapRoof);
			cv::Mat matLapRoofNoise(512, 512, CV_8UC1, lapRoofNoise);
			cv::Mat matSobRoof(512, 512, CV_8UC1, sobRoof);
			cv::Mat matSobRoofNoise(512, 512, CV_8UC1, sobRoofNoise);

			cv::imshow("Roof", matRoof);
			cv::imshow("Laplacian on Roof", matLapRoof);
			cv::imshow("Laplacian on Noisy Roof", matLapRoofNoise);
			cv::imshow("Sobel on Roof", matSobRoof);
			cv::imshow("Sobel on Noisy Roof", matSobRoofNoise);

			cv::waitKey(0);
			cv::destroyAllWindows();
		}

		/** 3_b
		 *   
		 */
		else if (select == 6) 
		{
			// Apply Gaussian blur on original image
			unsigned char* x = gaussianBlur(imgReceipt, 686, 512, 3, 3);
			saveRawAndPng("3_b_1", 686, 512, x);
			showImgCV(x, 686, 512, "Gaussian(£m=3, £g=0) on Original Image");
			// Then apply laplacian filtering
			x = laplacian(x, 686, 512);
			saveRawAndPng("3_b_2", 686, 512, x);
			showImgCV(x, 686, 512, "Laplacian + Gaussian(£m=3, £g=0)");
			// Adding filtered image with original image, the original image is multiplied by 0.1.
			x = addImg(x, imgReceipt, 686, 512);
			saveRawAndPng("3_b_3", 686, 512, x);
			showImgCV(x, 686, 512, "Add filtered image with original image * 0.1");
			// Thresholding
			x = thersholding(x, 686, 512, 0.175);
			saveRawAndPng("3_b_4", 686, 512, x);
			showImgCV(x, 686, 512, "Thresholding by 0.175");
		}
	}
	delete[] imglena512, imgCat512, imgCatch, imgRain, imgUmbrella, imgRoof, imgRoofNoise, imgReceipt;
}
