#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

void popOutMenu()
{
	cout
		<< "0: Exit the program." << endl << endl

		<< "Question 1: Bit-Plane (20%)" << endl
		<< "-----------------------------------" << endl
		<< "1: 1_1 - Rebuild Hidden Pepper" << endl
		<< "2: 1_2 - Rebuild 8 Bit-plane Images" << endl << endl

		<< "Question 2: Grey Level Transformation (30%)" << endl
		<< "-----------------------------------" << endl
		<< "3: 2_1 - Power-Law Transformation" << endl
		<< "4: 2_2 - Piecewise-Linear Transformation" << endl << endl

		<< "Question 3: Histograms Equalization (50%)" << endl
		<< "-----------------------------------" << endl
		<< "5: 3_1 - Plot Histograms" << endl
		<< "6: 3_2 - Histogram Equalization" << endl 
		<< "7: 3_3 - Compare Histograms" << endl
		<< "8: 3_4 - Fix Tsukuba" << endl << endl

		<< endl;
}
//---------- Miscellaneous ----------//
unsigned char* readImg(const char* fileNameImg, int height, int width)
{
	int size = height * width;
	unsigned char* imgArr = new unsigned char[size];
	FILE* imgFile = fopen(fileNameImg, "rb");
	fread(imgArr, sizeof(unsigned char), size, imgFile);
	return imgArr;
}
void showImgCV(unsigned char* imgArr, int height, int width, std::string figName)
{
	cv::Mat matrix(height, width, CV_8UC1, imgArr);
	cv::imshow(figName, matrix);
	cv::waitKey(0);
	cv::destroyAllWindows();
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
vector<unsigned char*> bitPlaneSlicing(unsigned char* img, int height, int width)
/** @brief Split an image into 8 different bit planes.
 * 
 *  @param img: Array of an image.
 *  @param height
 *  @param width
 *  @return A vector consists of each bit plane arranged by k-th bit.
 */
{
	int size = height * width;
	vector<unsigned char*> bitPlanes;
	for (auto bit = 7; bit >= 0; bit--) {
		unsigned char* bitImg = new unsigned char[size];
		for (auto row = 0; row < height; row++) {
			for (auto col = 0; col < width; col++) {
				int index = row * width + col;
				if (img[index] >= pow(2, bit)) {
					bitImg[index] = 255;
					img[index] -= pow(2, bit);
				}
				else { bitImg[index] = 0; }
			}
		}
		bitPlanes.push_back(bitImg);
	}
	reverse(bitPlanes.begin(), bitPlanes.end());
	return bitPlanes;
}
unsigned char* concatBitPlanes(vector<unsigned char*> bitPlanes, int height, int width)
/** @brief Given a vector of bit planes, rebuild them
 *		   by adding values by 2^n for every bit plane.
 *  
 *  The function deal with the bit planes from the bottom,
 *  thus the bit planes should be in order in the vector.
 *  The computation is based on the index of vector, e.g.,
 *  the bit plane at vector[2] is recognized as the 2nd bit 
 *  plane, thus the value is converted as pow(2, 2).
 * 
 *  @param bitPlanes: A n-vector contains n bit planes.
 *  @param height
 *  @param width
 * 
 *  @return rebuildImg: Recovered image.
 */
{
	int size = height * width;
	int numPlanes = bitPlanes.size();
	unsigned char* rebuildImg = new unsigned char[size];

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int index = row * width + col;
			int rebuildIntensity = 0;
			for (auto power = 0; power < numPlanes; power++) {
				if (bitPlanes[power][index] != 0) { rebuildIntensity += pow(2, power); }
			}
			rebuildImg[index] = rebuildIntensity;
		}
	}
	return rebuildImg;
}
unsigned char* colorExpansion(unsigned char* srcArr, int height, int width, int srcBit, int desBit)
{
	int size = height * width;
	unsigned char* desArr = new unsigned char[size];

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int index = row * width + col;

			// Quantization
			int divisor = pow(2, srcBit) - 1;
			int numerator = pow(2, desBit) - 1;
			int maxQuantizedValue = round(numerator / divisor);
			desArr[index] = srcArr[index] * maxQuantizedValue;
		}
	}
	return desArr;
}
unsigned char* negativeEffect(unsigned char* srcArr, int height, int width)
{
	int size = height * width;
	unsigned char* desArr = new unsigned char[size];

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int index = row * width + col;
			desArr[index] = -srcArr[index] + 255;
		}
	}

	return desArr;
}
//---------- Question 2 ----------//
unsigned char* powerLawTransformation(unsigned char* srcArr, int height, int width, float gamma, float c = 1.0)
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
unsigned char* darkPiecewiseTransformation(unsigned char* srcArr, int height, int width)
{
	int size = height * width;
	unsigned char* desArr = new unsigned char[size];

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int index = row * width + col;
			if (srcArr[index] < 64) { desArr[index] = srcArr[index] * 2 + 32; }
			else if (srcArr[index] > 200) { desArr[index] = srcArr[index]; }
			else { desArr[index] = 0.495 * srcArr[index] + 128.84 ;} 
		}
	}

	return desArr;
}
unsigned char* brightPiecewiseTransformation(unsigned char* srcArr, int height, int width)
{
	int size = height * width;
	unsigned char* desArr = new unsigned char[size];

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int index = row * width + col;
			if (((float)srcArr[index] / 255) >= 0.5) { desArr[index] = (((float)srcArr[index] / 255) * 1.6 - 0.6) * 255; }
			else { desArr[index] = 255 * (((float)srcArr[index] / 255) * 0.4); }
		}
	}
	return desArr;
}
unsigned char* contrastPiecewiseTransformation(unsigned char* srcArr, int height, int width)
{
	int size = height * width;
	unsigned char* desArr = new unsigned char[size];

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int index = row * width + col;
			double normedPix = (double)srcArr[index] / 255; // [0, 1]

			if (normedPix <= 0.2) {
				desArr[index] = (int)(normedPix * 0.3 * 255);
			}
			else if ((normedPix > 0.2) && (normedPix <= 0.81)) { 
				desArr[index] = (int)((normedPix * 1.54 - 0.248) * 255); 
			}
			else { desArr[index] = (int)255; }
		}
	}
	return desArr;
}
//---------- Question 3 ----------//
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
vector<int> computeEqualizedHist(unsigned char* img, int height, int width, bool q3_4=false) 
{	
	int size = height * width;
	vector<int> srcHist = computeHist(img, height, width);
	vector<int> desHist(256, 0);

	int cumulation = 0;
	for (auto idx = 0; idx < srcHist.size(); idx++) {
		float newPosition = 0.0;
		if (srcHist[idx] == 0) { continue; }
		cumulation += srcHist[idx];
		if (idx == 0) { newPosition = 0; }
		else { 
			if (q3_4 == true) { newPosition = 240 * (float)cumulation / size; }
			else { newPosition = 255 * (float)cumulation / size; }
		}
		desHist[newPosition] = srcHist[idx];
	}
	return desHist;
}
cv::Mat plotHist(unsigned char* img, int height, int width, float scale=0.8, bool equalizedHist=false, bool showFig=false)
{
	int size = height * width;
	unsigned char* outputImg = new unsigned char[size];

	vector<int> hist;
	if (equalizedHist == true) { hist = computeEqualizedHist(img, height, width); }
	else{ hist = computeHist(img, height, width); }

	int maxValue = *max_element(hist.begin(), hist.end()); // Max value in hist
	int maxHeight = round(height * scale); // The highest bar in hist

	float barScale = (float)maxHeight / maxValue; // Everybody multiplied by this will be scaled to fit in the figure
	cv::Mat3b matOutput = cv::Mat3b(height, width, cv::Vec3b(0, 0, 0));

	// Remember that openCV coordinates from top
	// Loop over hist vector
	for (auto xAxis = 0; xAxis < hist.size(); xAxis++) { 
		int barHeight = hist[xAxis] * barScale;
		cv::Point pt1(xAxis * 2, height - 1 - barHeight); 
		cv::Point pt2(xAxis * 2 + 1, height - 1);
		cv::rectangle(matOutput, pt1, pt2, (xAxis % 2) ? cv::Scalar(0, 100, 255) : cv::Scalar(0, 0, 255), cv::FILLED);
	}
	if (showFig == true) {
		cv::imshow("Histogram", matOutput);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
	return matOutput;
}
vector<int> getTransformVec(unsigned char* img, int height, int width)
{
	int size = height * width;
	vector<int> srcHist = computeHist(img, height, width);
	vector<int> transform(256, 0);

	int cumulation = 0;
	for (auto idx = 0; idx < srcHist.size(); idx++) {
		float newPosition = 0.0;
		if (srcHist[idx] == 0) { continue; }
		cumulation += srcHist[idx];
		if (idx == 0) { newPosition = 0; }
		else { newPosition = 255 * (float)cumulation / size; }
		transform[idx] = round(newPosition);
	}
	return transform;
}
unsigned char* plotFigEqualizedHist(unsigned char* img, int height, int width, bool show=false)
{
	int size = height * width;
	unsigned char* desArr = new unsigned char[size];

	vector<int> equalizedHist = computeEqualizedHist(img, height, width);
	vector<int> transformVec = getTransformVec(img, height, width);

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int idx = width * row + col;
			desArr[idx] = transformVec[img[idx]];
		}
	}
	if (show == true) { showImgCV(desArr, height, width, "Debug Mode"); }
	return desArr;
}
unsigned char* fixOverExposure(unsigned char* srcArr, int height, int width, bool q3_4=false)
{
	int size = height * width;
	unsigned char* desArr = new unsigned char[size];

	vector<int> equalizedHist = computeEqualizedHist(srcArr, height, width, q3_4);
	vector<int> transformVec = getTransformVec(srcArr, height, width);

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int idx = width * row + col;
			desArr[idx] = transformVec[srcArr[idx]];
		}
	}
	return desArr;
}


int main()
{
	//---------- Initializing ----------//
	unsigned char* imgBaboon256 = readImg("baboon_with_pepper_256.raw", 256, 256);
	unsigned char* imgCatDark512 = readImg("cat_dark_512.raw", 512, 512);
	unsigned char* imgCatBright512 = readImg("cat_bright_512.raw", 512, 512);
	unsigned char* imgCatLowContrast512 = readImg("cat_low_contrast_512.raw", 512, 512);
	unsigned char* imgTsukuba = readImg("tsukuba_683x512.raw", 512, 683);

	cv::Mat matCatDark512(512, 512, CV_8UC1, imgCatDark512);
	cv::Mat matCatBright512(512, 512, CV_8UC1, imgCatBright512);
	cv::Mat matCatLowContrast512(512, 512, CV_8UC1, imgCatLowContrast512);
	//---------- Menu ----------//
	popOutMenu();
	while (true)
	{
		cout << "Enter the Question Number to Show Answer:" << endl;
		int select = 0;
		cin >> select;

		if (select == 0) { break; }
		// 1_1 - Rebuild Hidden Pepper
		else if (select == 1) {
			vector<unsigned char*> baboonBitPlanes = bitPlaneSlicing(imgBaboon256, 256, 256);

			// Get first 4 bit planes
			vector<unsigned char*> pepperBitPlanes(baboonBitPlanes.begin(), baboonBitPlanes.begin() + 4);
			unsigned char* rebuildPepper = concatBitPlanes(pepperBitPlanes, 256, 256);

			// 4-bit to 8-bit
			rebuildPepper = colorExpansion(rebuildPepper, 256, 256, 4, 8);
			saveRawAndPng("pepper_hidden_256", 256, 256, rebuildPepper);
			showImgCV(rebuildPepper, 256, 256, "pepper_hidden_256.raw");
		}
		// 1_2 - Rebuild 8 Bit-plane Images
		else if (select == 2) {
			unsigned char* bitPlane0 = readImg("d_512.raw", 512, 512);
			unsigned char* bitPlane1 = readImg("f_512.raw", 512, 512);
			unsigned char* bitPlane2 = readImg("g_512.raw", 512, 512);
			unsigned char* bitPlane3 = readImg("e_512.raw", 512, 512);
			unsigned char* bitPlane4 = readImg("c_512.raw", 512, 512);
			unsigned char* bitPlane5 = readImg("a_512.raw", 512, 512); // negative
			unsigned char* bitPlane6 = readImg("h_512.raw", 512, 512); // negative
			unsigned char* bitPlane7 = readImg("b_512.raw", 512, 512);
			
			// Fix Negative Films
			bitPlane2 = negativeEffect(bitPlane2, 512, 512);
			bitPlane3 = negativeEffect(bitPlane3, 512, 512);
			bitPlane5 = negativeEffect(bitPlane5, 512, 512);
			bitPlane6 = negativeEffect(bitPlane6, 512, 512);

			// A vector used to store bit planes
			vector<unsigned char*> pancakeBitPlanes = {
				bitPlane0, bitPlane1, bitPlane2, bitPlane3,
				bitPlane4, bitPlane5, bitPlane6, bitPlane7
			};

			unsigned char* rebuildPancake = concatBitPlanes(pancakeBitPlanes, 512, 512);
			saveRawAndPng("pancake", 512, 512, rebuildPancake);
			showImgCV(rebuildPancake, 512, 512, "pancake.raw");

			delete[] bitPlane0, bitPlane1, bitPlane2, bitPlane3;
			delete[] bitPlane4, bitPlane5, bitPlane6, bitPlane7;
			delete[] rebuildPancake;
		}
		// 2_1 - Power-Law Transformation
		else if (select == 3) {
			//---------- Dark Cat ----------//
			float dark_gamma_1 = 0.5, dark_gamma_2 = 0.6;
			float dark_c_1 = 1, dark_c_2 = 1.5;

			unsigned char* catDarkSet1 = powerLawTransformation(imgCatDark512, 512, 512, dark_gamma_1, dark_c_1);
			unsigned char* catDarkSet2 = powerLawTransformation(imgCatDark512, 512, 512, dark_gamma_2, dark_c_2); // Best

			// Transform Arrays to cv::Mat
			cv::Mat matCatDarkSet1(512, 512, CV_8UC1, catDarkSet1);
			cv::Mat matCatDarkSet2(512, 512, CV_8UC1, catDarkSet2);

			saveRawAndPng("2_1_dark_1", 512, 512, catDarkSet1);
			saveRawAndPng("2_1_dark_2", 512, 512, catDarkSet2);

			// Show Images
			cv::imshow("Dark: gamma_1 = 0.5, c_1 = 1", matCatDarkSet1);
			cv::imshow("Dark: gamma_2 = 0.6, c_2 = 1.5", matCatDarkSet2);
			cv::waitKey(0);
			cv::destroyAllWindows();

			//---------- Bright Cat ----------//
			float bright_gamma_1 = 7.5, bright_gamma_2 = 2;
			float bright_c_1 = 0.75, bright_c_2 = 0.75;

			unsigned char* catBrightSet1 = powerLawTransformation(imgCatBright512, 512, 512, bright_gamma_1, bright_c_1);
			unsigned char* catBrightSet2 = powerLawTransformation(imgCatBright512, 512, 512, bright_gamma_2, bright_c_2); // Best

			saveRawAndPng("2_1_bright_1", 512, 512, catBrightSet1);
			saveRawAndPng("2_1_bright_2", 512, 512, catBrightSet2);

			// Transform Arrays to cv::Mat
			cv::Mat matCatBrightSet1(512, 512, CV_8UC1, catBrightSet1);
			cv::Mat matCatBrightSet2(512, 512, CV_8UC1, catBrightSet2);

			// Show Images
			cv::imshow("Bright: gamma_1 = 7.5, c_1 = 0.75", matCatBrightSet1);
			cv::imshow("Bright: gamma_2 = 2, c_2 = 0.75", matCatBrightSet2);
			cv::waitKey(0);
			cv::destroyAllWindows();

			//---------- Low Contrast Cat ----------//
			float low_con_gamma_1 = 2, low_con_gamma_2 = 1.5;
			float low_con_c_1 = 1.5, low_con_c_2 = 1.25;

			unsigned char* catLowContrastSet1 = powerLawTransformation(imgCatLowContrast512, 512, 512, low_con_gamma_1, low_con_c_1); // Best
			unsigned char* catLowContrastSet2 = powerLawTransformation(imgCatLowContrast512, 512, 512, low_con_gamma_2, low_con_c_2);

			saveRawAndPng("2_1_lc_1", 512, 512, catLowContrastSet1);
			saveRawAndPng("2_1_lc_2", 512, 512, catLowContrastSet2);

			// Transform Arrays to cv::Mat
			cv::Mat matCatLowContrastSet1(512, 512, CV_8UC1, catLowContrastSet1);
			cv::Mat matCatLowContrastSet2(512, 512, CV_8UC1, catLowContrastSet2);

			// Show Images
			cv::imshow("Low Contrast: gamma_1 = 2, c_1 = 1.5", matCatLowContrastSet1);
			cv::imshow("Low Contrast: gamma_2 = 1.5, c_2 = 1.25", matCatLowContrastSet2);
			cv::waitKey(0);
			cv::destroyAllWindows();

			delete[] catDarkSet1, catDarkSet2;
			delete[] catBrightSet1, catBrightSet2;
			delete[] catLowContrastSet1, catLowContrastSet2;
		}
		// 2_2 - Piecewise-Linear Transformation
		else if (select == 4) {
			//---------- Dark Cat ----------//
			unsigned char* darkExpandImg = darkPiecewiseTransformation(imgCatDark512, 512, 512);
			saveRawAndPng("2_2_dark", 512, 512, darkExpandImg);

			// Transform Arrays to cv::Mat
			cv::Mat matCatDark512(512, 512, CV_8UC1, imgCatDark512);
			cv::Mat matDarkExpandImg(512, 512, CV_8UC1, darkExpandImg);

			// Show Images
			cv::imshow("Dark: Original", matCatDark512);
			cv::imshow("Dark: Result", matDarkExpandImg);
			cv::waitKey(0);
			cv::destroyAllWindows();

			//---------- Bright Cat ----------//
			unsigned char* brightExpandImg = brightPiecewiseTransformation(imgCatBright512, 512, 512);
			saveRawAndPng("2_2_bright", 512, 512, brightExpandImg);

			// Transform Arrays to cv::Mat
			cv::Mat matCatBright512(512, 512, CV_8UC1, imgCatBright512);
			cv::Mat matBrightExpandImg(512, 512, CV_8UC1, brightExpandImg);

			// Show Images
			cv::imshow("Bright: Original", matCatBright512);
			cv::imshow("Bright: Result", matBrightExpandImg);
			cv::waitKey(0);
			cv::destroyAllWindows();

			//---------- Contrast Cat ----------//
			unsigned char* contrastExpandImg = contrastPiecewiseTransformation(imgCatLowContrast512, 512, 512);
			saveRawAndPng("2_2_contrast", 512, 512, contrastExpandImg);

			// Transform Arrays to cv::Mat
			cv::Mat matCatContrast512(512, 512, CV_8UC1, imgCatLowContrast512);
			cv::Mat matLowContrastImg(512, 512, CV_8UC1, contrastExpandImg);

			// Show Images
			cv::imshow("Low Contrast: Original", matCatContrast512);
			cv::imshow("New Contrast: Result", matLowContrastImg);
			cv::waitKey(0);
			cv::destroyAllWindows();

			delete[] darkExpandImg, brightExpandImg, contrastExpandImg;
		}
		// 3_1 - Plot Histograms
		else if (select == 5) {
			cv::Mat matHistCarDark = plotHist(imgCatDark512, 512, 512, 0.8);
			cv::Mat matHistCatBright = plotHist(imgCatBright512, 512, 512, 0.8);
			cv::Mat matHistCatLowContrast = plotHist(imgCatLowContrast512, 512, 512, 0.8);

			// Show Images
			cv::imshow("Cat Dark", matHistCarDark);
			cv::imshow("Cat Bright", matHistCatBright);
			cv::imshow("Cat Low Contrast", matHistCatLowContrast);
			cv::waitKey(0);
			cv::destroyAllWindows();

			cv::imwrite("3_1_hist_cat_dark.png", matHistCarDark);
			cv::imwrite("3_1_hist_cat_bright.png", matHistCatBright);
			cv::imwrite("3_1_hist_cat_low_contrast.png", matHistCatLowContrast);
			cout << "Images saved!" << endl << endl;
		}
		// 3_2 - Histogram Equalization
		else if (select == 6) {
			//---------- Dark Cat ----------//
			cv::Mat matHistCarDark = plotHist(imgCatDark512, 512, 512, 0.8, false);
			cv::Mat matEqualizedHistCarDark = plotHist(imgCatDark512, 512, 512, 0.8, true);
			unsigned char* equalizedCatDark = plotFigEqualizedHist(imgCatDark512, 512, 512);
			cv::Mat matEqualizedCatDark(512, 512, CV_8UC1, equalizedCatDark);

			//vector<int> x = getTransformVec(equalizedCatDark, 512, 512);
			//for (auto i = 0; i != x.size();i++) {
			//	cout << i << "," << x[i] << endl;
			//}

			// Show Images
			cv::imshow("Original Histogram", matHistCarDark);
			cv::imshow("Equalized Histogram", matEqualizedHistCarDark);
			cv::imshow("Original Figure", matCatDark512);
			cv::imshow("Equalized Figure", matEqualizedCatDark);
			cv::waitKey(0);
			cv::destroyAllWindows();

			// Save Images
			cv::imwrite("3_2_1_equalized_hist_cat_dark.png", matEqualizedHistCarDark);
			cv::imwrite("3_2_1_equalized_cat_dark.png", matEqualizedCatDark);
			cout << "Images saved!" << endl << endl;

			//---------- Bright Cat ----------//
			cv::Mat matHistCatBright = plotHist(imgCatBright512, 512, 512, 0.8, false);
			cv::Mat matEqualizedHistCatBright = plotHist(imgCatBright512, 512, 512, 0.8, true);
			unsigned char* equalizedCatBright = plotFigEqualizedHist(imgCatBright512, 512, 512);
			cv::Mat matEqualizedCatBright(512, 512, CV_8UC1, equalizedCatBright);

			// Show Images
			cv::imshow("Original Histogram", matHistCatBright);
			cv::imshow("Equalized Histogram", matEqualizedHistCatBright);
			cv::imshow("Original Figure", matCatBright512);
			cv::imshow("Equalized Figure", matEqualizedCatBright);
			cv::waitKey(0);
			cv::destroyAllWindows();

			// Save Images
			cv::imwrite("3_2_2_equalized_hist_cat_bright.png", matEqualizedHistCatBright);
			cv::imwrite("3_2_2_equalized_cat_bright.png", matEqualizedCatBright);
			cout << "Images saved!" << endl << endl;

			//---------- Low Contrast Cat ----------//
			cv::Mat matHistCatLowContrast = plotHist(imgCatLowContrast512, 512, 512, 0.8, false);
			cv::Mat matEqualizedHistCatLowContrast = plotHist(imgCatLowContrast512, 512, 512, 0.8, true);
			unsigned char* equalizedCatLowContrast = plotFigEqualizedHist(imgCatLowContrast512, 512, 512);
			cv::Mat matEqualizedCatLowContrast(512, 512, CV_8UC1, equalizedCatLowContrast);

			// Show Images
			cv::imshow("Original Histogram", matHistCatLowContrast);
			cv::imshow("Equalized Histogram", matEqualizedHistCatLowContrast);
			cv::imshow("Original Figure", matCatLowContrast512);
			cv::imshow("Equalized Figure", matEqualizedCatLowContrast);
			cv::waitKey(0);
			cv::destroyAllWindows();

			// Save Images
			cv::imwrite("3_2_3_equalized_hist_cat_low_contrast.png", matEqualizedHistCatLowContrast);
			cv::imwrite("3_2_3_equalized_cat_low_contrast.png", matEqualizedCatLowContrast);
			cout << "Images saved!" << endl << endl;
		}
		// 3_3 - Histograms of 2_1 & 2_2
		else if (select == 7) {
			//---------- 2_1's Figures ----------//
			unsigned char* myCatDark = powerLawTransformation(imgCatDark512, 512, 512, 0.6, 1.5);
			unsigned char* myCatBright = powerLawTransformation(imgCatBright512, 512, 512, 2, 0.75);
			unsigned char* myCatLowContrast = powerLawTransformation(imgCatLowContrast512, 512, 512, 2, 1.5);

			cv::Mat matHistMyCarDark = plotHist(myCatDark, 512, 512, 0.8);
			cv::Mat matHistMyCatBright = plotHist(myCatBright, 512, 512, 0.8);
			cv::Mat matHistMyCatLowContrast = plotHist(myCatLowContrast, 512, 512, 0.8);

			// Show Images
			cv::imshow("My Power-Law Cat Dark", matHistMyCarDark);
			cv::imshow("My Power-Law Cat Bright", matHistMyCatBright);
			cv::imshow("My Power-Law Cat Low Contrast", matHistMyCatLowContrast);
			cv::waitKey(0);
			cv::destroyAllWindows();

			cv::imwrite("3_3_1_hist_PL_cat_dark.png", matHistMyCarDark);
			cv::imwrite("3_3_1_hist_PL_cat_bright.png", matHistMyCatBright);
			cv::imwrite("3_3_1_hist_PL_cat_low_contrast.png", matHistMyCatLowContrast);
			cout << "Images saved!" << endl << endl;

			delete[] myCatDark, myCatBright, myCatLowContrast;

			//---------- 2_2's Figures ----------//
			unsigned char* darkExpandImg = darkPiecewiseTransformation(imgCatDark512, 512, 512);
			unsigned char* brightExpandImg = brightPiecewiseTransformation(imgCatBright512, 512, 512);
			unsigned char* contrastExpandImg = contrastPiecewiseTransformation(imgCatLowContrast512, 512, 512);

			cv::Mat matHistDarkExpand = plotHist(darkExpandImg, 512, 512, 0.8);
			cv::Mat matHistBrightExpand = plotHist(brightExpandImg, 512, 512, 0.8);
			cv::Mat matHistContrastExpand = plotHist(contrastExpandImg, 512, 512, 0.8);

			// Show Images
			cv::imshow("My Piecewise Cat Dark", matHistDarkExpand);
			cv::imshow("My Piecewise Cat Bright", matHistBrightExpand);
			cv::imshow("My Piecewise Cat Low Contrast", matHistContrastExpand);
			cv::waitKey(0);
			cv::destroyAllWindows();

			cv::imwrite("3_3_2_hist_piece_cat_dark.png", matHistDarkExpand);
			cv::imwrite("3_3_2_hist_piece_cat_bright.png", matHistBrightExpand);
			cv::imwrite("3_3_2_hist_piece_cat_low_contrast.png", matHistContrastExpand);
			cout << "Images saved!" << endl << endl;

			delete[] darkExpandImg, brightExpandImg, contrastExpandImg;
		}
		// 3_4 - Fix Over-Exposured
		else if (select == 8) {
			unsigned char* tsukuba = plotFigEqualizedHist(imgTsukuba, 512, 683, false);
			tsukuba = powerLawTransformation(tsukuba, 512, 683, 2.5, 1);
			//tsukuba = fixOverExposure(tsukuba, 512, 683, true);
			showImgCV(tsukuba, 512, 683, "Tsukuba");
			saveRawAndPng("tsukuba", 512, 683, tsukuba);
		}
		else if (select == 9) {
			vector<int> x = getTransformVec(imgCatBright512, 512, 512);
			for (auto i = 0; i != x.size();i++) {
				cout << i << "," << x[i] << endl;
			}
		}
		else { cout << "Try again!" << endl << endl; }
	}
	//---------- Release Memory ----------//
	delete[] imgBaboon256;
	delete[] imgCatDark512;
	delete[] imgCatBright512;
	delete[] imgCatLowContrast512;
	delete[] imgTsukuba;

	return 0;
}