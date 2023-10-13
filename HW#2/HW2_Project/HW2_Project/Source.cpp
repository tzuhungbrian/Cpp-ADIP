#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <math.h>
#include <vector>

using namespace std;

void popOutMenu()
{
	cout
		<< "0: Exit the program." << endl << endl

		<< "Question 1: Resizing" << endl
		<< "-----------------------------------" << endl
		<< "1: 1_1 - Row-Column Replication" << endl
		<< "2: 1_2 - Three Zooming Algorithms" << endl
		<< "3: 1_3 - Row-Column Deletion" << endl
		<< "4: 1_4 - Resize by Every Algorithms with Different Ratios" << endl << endl

		<< "Question 2: MSE and PSNR" << endl
		<< "-----------------------------------" << endl
		<< "5: 2_1 - Compute MSE and PSNR between image 1_1 and lena_512.raw" << endl
		<< "6: 2_2 - Compute MSE and PSNR between source image and resized image" << endl << endl

		<< "Quenstion 3: Transmission" << endl
		<< "-----------------------------------" << endl
		<< "7: 3_1 - Reduce Spatial Resolution and Compute MSE and PSNR" << endl
		<< "8: 3_2 - Reduce Color Resolution and Compute MSE and PSNR" << endl
		<< "9: 3_3 - Mix Two Methods to Reduce Image Size" << endl
		<< endl;
}

// Quenstion 1

unsigned char* rcReplicate(unsigned char* imgArr, int height, int width, int propRow, int propCol)
{
	int newHeight = propRow * height;
	int newWidth = propCol * width;
	int newSize = newHeight * newWidth;
	unsigned char* newArr = new unsigned char[newSize];

	// Duplicate rows
	if (propRow != 1) {
		for (auto row = 0; row != newHeight; row += propRow) {
			for (auto col = 0; col != width; ++col) {
				for (auto dupRow = 0; dupRow != propRow; ++dupRow) {
					int originIdx = row / propRow * width + col;
					int index = row * newWidth + col;
					newArr[index + dupRow * newWidth] = imgArr[originIdx];
				}
			}
		}
	}

	// Duplicate columns
	if (propCol != 1) {
		for (auto row = 0; row != newHeight; ++row) {
			for (auto col = 0; col != newWidth; col += propCol) {
				for (auto dupCol = 0; dupCol != propCol; ++dupCol) {
					int originIdx = row / propRow * width + col / propCol;
					int index = row * newWidth + col;
					newArr[index + dupCol] = imgArr[originIdx];
				}
			}
		}
	}

	return newArr;
}

unsigned char* rcDeletion(unsigned char* srcArr, int height, int width, int propRow, int propCol)
{
	int desHeight = height / propRow;
	int desWidth = width / propCol;
	int desSize = desHeight * desWidth;
	unsigned char* desArr = new unsigned char[desSize];

	// Column Deletion
	if (propCol != 1) {
		for (auto row = 0; row != desHeight; ++row) {
			for (auto col = 0; col != desWidth; ++col) {
				int srcIdx = row * width * propRow + col * propCol;
				int desIdx = row * desWidth + col;
				desArr[desIdx] = srcArr[srcIdx];
			}
		}
	}

	// Row Deletion
	if (propRow != 1) {
		for (auto row = 0; row != desHeight; ++row) {
			for (auto col = 0; col != desWidth; ++col) {
				int srcIdx = row * propRow * width + col * propCol;
				int desIdx = row * desWidth + col;
				desArr[desIdx] = srcArr[srcIdx];
			}
		}
	}
	return desArr;
}

unsigned char* nearestNeighbor(unsigned char* srcArr, int height, int width, float scale)
{
	int desHeight = scale * height, desWidth = scale * width;
	int desSize = desHeight * desWidth;
	unsigned char* desArr = new unsigned char[desSize];

	for (auto row = 0; row != desHeight; ++row) {
		int srcRow = round(row / scale);
		for (auto col = 0; col != desWidth; ++col) {
			int srcIdx = srcRow * width + round(col / scale);
			int desIdx = desWidth * row + col;

			desArr[desIdx] = srcArr[srcIdx];
		}
	}
	return desArr;
}

unsigned char* bilinear(unsigned char* srcArr, int height, int width, float scale)
{
	int desHeight = scale * height, desWidth = scale * width;
	int desSize = desHeight * desWidth;
	unsigned char* desArr = new unsigned char[desSize];

	if (scale >= 1) {
		for (auto row = 0; row != desHeight; ++row) {
			int srcRow = int(row / scale);
			for (auto col = 0; col != desWidth; ++col) {
				int srcCol = int(col / scale);
				int srcIdx = srcRow * width + col / scale;

				int p1 = srcArr[srcIdx], p2 = srcArr[srcIdx + 1]; // Top left & right
				int p3 = srcArr[srcIdx + width], p4 = srcArr[srcIdx + width + 1]; // Bot left & right
				
				if (srcCol == width - 1) { p2 = srcArr[srcIdx], p4 = srcArr[srcIdx + width]; }
				//---------- Compute Interpolation ----------//
				int x1 = col - round(srcCol * scale), x2 = round((srcCol + 1) * scale) - col;
				float totalX = x1 + x2;
				float px1 = x1 / totalX, px2 = x2 / totalX;
				int y1 = row - round(srcRow * scale), y2 = round((srcRow + width) * scale) - row;
				float totalY = y1 + y2;
				float py1 = y1 / totalY, py2 = y2 / totalY;

				int desIdx = row * desWidth + col;
				desArr[desIdx] = (p1 * px2 + p2 * px1) * py2 + (p3 * px2 + p4 * px1) * py1;
			}
		}
	}
	else {
		for (auto row = 0; row != desHeight; ++row) {
			int srcRow = int(row / scale);
			for (auto col = 0; col != desWidth; ++col) {
				int srcCol = int(col / scale);
				int srcIdx = srcRow * width + col / scale;
				int desIdx = row * desWidth + col;

				int p1 = srcArr[srcIdx], p2 = srcArr[srcIdx + 1];
				int p3 = srcArr[srcIdx + width], p4 = srcArr[srcIdx + width + 1];
				if (srcIdx + 1 > 255) { p2 = p1, p4 = p3; }
				float meanP = (p1 + p2 + p3 + p4) / 4;

				desArr[desIdx] = meanP;
			}
		}
	}

	return desArr;
}

int cubicInterpolateArr(int p[4], float interpolateX)
{
	return int(p[1] + 0.5 * interpolateX * (p[2] - p[0] + interpolateX * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + interpolateX * (3.0 * (p[1] - p[2]) + p[3] - p[0]))));
}

int bicubicInterpolateArr(int p[4][4], float interploateX, float interploateY)
{
	int arr[4];
	for (auto i = 0; i < 4; ++i) {
		arr[i] = cubicInterpolateArr(p[i], interploateX);
	}
	return cubicInterpolateArr(arr, interploateY);
}

unsigned char* bicubic(unsigned char* srcArr, int height, int width, float scale, bool print=false)
{
	int desHeight = scale * height, desWidth = scale * width;
	int desSize = desHeight * desWidth;
	unsigned char* desArr = new unsigned char[desSize];

	for (auto row = 0; row < desHeight; row++) {
		int srcRow = floor(row / scale);
		for (auto col = 0; col < desWidth; col++) {
			int srcCol = floor(col / scale);

			// Create Array for Bicubic Interploation
			int pointArr[4][4];
			for (auto arrRow = 0; arrRow < 4; arrRow++) {
				int sampRow = srcRow - 1 + arrRow;
				if (sampRow >= height - 1) { sampRow = height - 1; } // Bottom Boundary Case
				for (auto arrCol = 0; arrCol < 4; arrCol++) {
					int sampCol = srcCol - 1 + arrCol;
					if (sampCol >= width - 1) { sampCol = width - 1; } // Right Boundary Case

					int sampIdx = sampRow * width + sampCol;
					if (sampIdx <= 0) { sampIdx = 0;} // Prevent from out-of-index error

					pointArr[arrRow][arrCol] = srcArr[sampIdx];
				}
			}

			// Calculate interplolate x
			int leftBound = floor(srcCol * scale);
			int rightBound = floor((srcCol + 1) * scale);
			float x;
			if (rightBound - leftBound == 0) { x = 0; }
			else { x = float(col - leftBound) / (rightBound - leftBound); }

			// Calculate interplolate y
			int upperBound = floor(srcRow * scale);
			int lowerBound = floor((srcRow + 1) * scale);
			float y;
			if (upperBound - lowerBound == 0) { y = 0; }
			else { y = float(row - upperBound) / (lowerBound - upperBound); }

			int desIdx = row * desWidth + col;
			desArr[desIdx] = bicubicInterpolateArr(pointArr, x, y);
		}
	}
	return desArr;
}

// Question 2

float mse(unsigned char* cleanImg, unsigned char* noisyImg, int height, int width, bool print=true) 
{
	int size = height * width;
	float pixelScore = 0;

	for (auto row = 0; row != height; ++row) {
		for (auto col = 0; col != width; ++col) {
			int index = row * width + col;
			pixelScore += pow((cleanImg[index] - noisyImg[index]), 2);
		}
	}
	float mseScore = pixelScore / size;
	if (print == true) { cout << "MSE Score is: " << mseScore << endl; }

	return mseScore;
}

float psnr(unsigned char* cleanImg, unsigned char* noisyImg, int height, int width, bool print=true)
{
	float mseScore = mse(cleanImg, noisyImg, height, width, print);
	int maxIntensity = 255; // 8-bit Image 2^8-1
	float psnrScore = 20 * log10((maxIntensity) / (sqrt(mseScore)));
	if (print == true) {
		cout << "PSNR is: " << psnrScore << "dB" << endl;
	}
	return psnrScore;
}

// Question 3

unsigned char* colorQuantization(unsigned char* srcArr, int height, int width, int bit)
{
	if (bit > 8 || bit < 1) { throw invalid_argument("Argument \"bit\" should be [1, 8]."); }

	int size = height * width;
	unsigned char* desArr = new unsigned char[size];

	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			int index = row * width + col;

			// Quantization
			int numColors = pow(2, bit);
			int divisor = 256 / numColors;
			int maxQuantizedValue = 255 / divisor;
			desArr[index] = ((srcArr[index] / divisor) * 255) / maxQuantizedValue;
		}
	}
	return desArr;
}

// Misc

void showImgCV(unsigned char* imgArr, int height, int width, std::string figName)
{
	cv::Mat matrix(height, width, CV_8UC1, imgArr);
	cv::imshow(figName, matrix);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

void saveRawImg(const char* saveName, int imgSize, unsigned char* saveImgArr)
{
	FILE* saveFile = fopen(saveName, "wb");
	fwrite(saveImgArr, 1, imgSize, saveFile);
	std::cout << "Image Saved!" << std::endl;
	fclose(saveFile);
}

int main()
{
	//---------- Initialize Parameters ----------//
	char lena256Img[] = "lena_256.raw", lena512Img[] = "lena_512.raw", lena512Blurred[] = "lena_512_blurred.raw";
	int lena256H = 256, lena256W = 256;
	int lena512H = 512, lena512W = 512;

	int lena256Size = lena256H * lena256W;
	int lena512Size = lena512H * lena512W;

	unsigned char* imgLena256 = new unsigned char[lena256Size];
	unsigned char* imgLena512 = new unsigned char[lena512Size];
	unsigned char* imgLena512Blurred = new unsigned char[lena512Size];

	//---------- Read Images into Array ----------//
	FILE* lena256File = fopen(lena256Img, "rb");
	FILE* lena512File = fopen(lena512Img, "rb");
	FILE* lena512BlurredFile = fopen(lena512Blurred, "rb");

	fread(imgLena256, sizeof(unsigned char), lena256Size, lena256File);
	fread(imgLena512, sizeof(unsigned char), lena512Size, lena512File);
	fread(imgLena512Blurred, sizeof(unsigned char), lena512Size, lena512BlurredFile);

	//---------- Menu ----------//
	popOutMenu();

	while (true) {
		cout << "Enter the Question Number to Show Answer:" << endl;
		int select = 0;
		cin >> select;

		//---------- Homeworks ----------//

		if (select == 0) { break; }
		// 1_1 Row-Column Replication (2, 1)
		else if (select == 1) {
			int resampleRow = 2, resampleCol = 2; // Resample proportion

			unsigned char* resizedImg = rcReplicate(imgLena256, lena256H, lena256W, resampleRow, resampleCol);

			// Transform Arrays to cv::Mat
			cv::Mat matResizedImg(lena256H * 2, lena256W * 2, CV_8UC1, resizedImg);
			cv::Mat matLena512(512, 512, CV_8UC1, imgLena512);
			
			// Save Image
			cv::imwrite("1_1.png", matResizedImg);
			cout << "Image(s) has been saved!" << endl;

			// Show Images
			cv::imshow("1_1: Row-Column Replication", matResizedImg);
			cv::imshow("1_1: Original 512 Image", matLena512);
			cv::waitKey(0);
			cv::destroyAllWindows();

			delete[] resizedImg;
		}

		// 1_2 Nearest Neighbors, Bilinear, Bicubic
		else if (select == 2) {
			float scale = 3.25;

			// 1) Nearest Neighbors
			unsigned char* nnImg = nearestNeighbor(imgLena256, lena256H, lena256W, scale);

			// 2) Bilinear
			unsigned char* bilinearImg = bilinear(imgLena256, lena256H, lena256W, scale);
			
			// 3) Bicubic
			unsigned char* bicubicImg = bicubic(imgLena256, lena256H, lena256W, scale);

			// Transform Arrays to cv::Mat
			cv::Mat matNN(lena256H * scale, lena256W * scale, CV_8UC1, nnImg);
			cv::Mat matBL(lena256H * scale, lena256W * scale, CV_8UC1, bilinearImg);
			cv::Mat matBC(lena256H * scale, lena256W * scale, CV_8UC1, bicubicImg);

			// Save Image
			cv::imwrite("1_2_NN.png", matNN);
			cv::imwrite("1_2_BL.png", matBL);
			cv::imwrite("1_2_BC.png", matBC);
			cout << "Image(s) has been saved!" << endl;

			// Show Images
			cv::imshow("1_2_1: Nearest Neighbors", matNN);
			cv::imshow("1_2_1: Bilinear", matBL);
			cv::imshow("1_2_1: Bicubic", matBC);
			cv::waitKey(0);
			cv::destroyAllWindows();

			delete[] bilinearImg;
			delete[] nnImg;
			delete[] bicubicImg;
		}

		// 1_3 Row-Column Deletion
		else if (select == 3) {
			int resampleRow = 2, resampleCol = 2;
			unsigned char* rcDeletionImg = rcDeletion(imgLena512, lena512H, lena512W, resampleRow, resampleCol);
			unsigned char* rcdBlurred = rcDeletion(imgLena512Blurred, lena512H, lena512W, resampleRow, resampleCol);
			//showImgCV(rcDeletionImg, lena512H / resampleRow, lena512W / resampleCol, "1_3: Row-Column Deletion");
			//showImgCV(rcdBlurred, lena512H / resampleRow, lena512W / resampleCol, "1_3: Blurred Row-Column Deletion");
			//showImgCV(imgLena256, lena256H , lena256W , "1_3: Original Image");

			// Save Image
			cv::Mat matRCD(lena512H / resampleRow, lena512W / resampleCol, CV_8UC1, rcDeletionImg);
			cv::Mat matBlurredRCD(lena512H / resampleRow, lena512W / resampleCol, CV_8UC1, rcdBlurred);
			cv::Mat matOriginal(lena256H, lena256W, CV_8UC1, imgLena256);

			cv::imwrite("1_3_Blurred.png", matBlurredRCD);
			cv::imwrite("1_3.png", matRCD);
			cout << "Image(s) has been saved!" << endl;

			// Show Images
			cv::imshow("1_3: Row-Column Deletion", matRCD);
			cv::imshow("1_3: Blurred Row-Column Deletion", matBlurredRCD);
			cv::imshow("Original Image", matOriginal);
			cv::waitKey(0);
			cv::destroyAllWindows();

			delete[] rcDeletionImg;
		}

		// 1_4 Resize by a) ^3.25v2.5, b) v2.5^3.25^, and c) ^1.3 with
		else if (select == 4) {
			int sideLen = 256 * .4; // Only used in b)

			// 1) Nearest Neighbors
			// a)
			unsigned char* img1_a = nearestNeighbor(imgLena256, lena256H, lena256W, 3.25);
			img1_a = nearestNeighbor(img1_a, lena256H * 3.25, lena256W * 3.25, 0.4);
			
			// b)
			unsigned char* img1_b = nearestNeighbor(imgLena256, lena256H, lena256W, .4);
			img1_b = nearestNeighbor(img1_b, sideLen, sideLen, 3.25);
			
			// c)
			unsigned char* img1_c = nearestNeighbor(imgLena256, lena256H, lena256W, 1.3);

			// Transform Arrays to cv::Mat
			cv::Mat mat1_a(lena256H * 3.25 * .4, lena256W * 3.25 * .4, CV_8UC1, img1_a);
			cv::Mat mat1_b(sideLen * 3.25, sideLen * 3.25, CV_8UC1, img1_b);
			cv::Mat mat1_c(lena256H * 1.3, lena256W * 1.3, CV_8UC1, img1_c);

			// Save Image
			cv::imwrite("1_4_NNa.png", mat1_a);
			cv::imwrite("1_4_NNb.png", mat1_b);
			cv::imwrite("1_4_NNc.png", mat1_c);
			cout << "Image(s) has been saved!" << endl;
			
			// Show Images
			cv::imshow("1_4: Nearest Neighbors ^3.25v2.5", mat1_a);
			cv::imshow("1_4: Nearest Neighbors v2.5^3.25", mat1_b);
			cv::imshow("1_4: Nearest Neighbors ^ 1.3", mat1_c);
			cv::waitKey(0);
			cv::destroyAllWindows();

			delete[] img1_a, delete[] img1_b, delete[] img1_c;

			// 2) Bilinear
			// a)
			unsigned char* img2_a = bilinear(imgLena256, lena256H, lena256W, 3.25);
			img2_a = bilinear(img2_a, lena256H * 3.25, lena256W * 3.25, 0.4);

			// b)
			unsigned char* img2_b = bilinear(imgLena256, lena256H, lena256W, .4);
			img2_b = bilinear(img2_b, sideLen, sideLen, 3.25);

			// c)
			unsigned char* img2_c = bilinear(imgLena256, lena256H, lena256W, 1.3);
			
			// Transform Arrays to cv::Mat
			cv::Mat mat2_a(lena256H * 3.25 * .4, lena256W * 3.25 * .4, CV_8UC1, img2_a);
			cv::Mat mat2_b(sideLen * 3.25, sideLen * 3.25, CV_8UC1, img2_b);
			cv::Mat mat2_c(lena256H * 1.3, lena256W * 1.3, CV_8UC1, img2_c);

			// Save Image
			cv::imwrite("1_4_BLa.png", mat2_a);
			cv::imwrite("1_4_BLb.png", mat2_b);
			cv::imwrite("1_4_BLc.png", mat2_c);
			cout << "Image(s) has been saved!" << endl;

			// Show Images
			cv::imshow("1_4: Bilinear ^3.25v2.5", mat2_a);
			cv::imshow("1_4: Bilinear v2.5^3.25", mat2_b);
			cv::imshow("1_4: Bilinear ^ 1.3", mat2_c);
			cv::waitKey(0);
			cv::destroyAllWindows();

			delete[] img2_a, delete[] img2_b, delete[] img2_c;

			// 3) Bicubic
			// a)
			unsigned char* img3_a = bicubic(imgLena256, lena256H, lena256W, 3.25); 
			// Buggy: Rough solution
			img3_a = bicubic(img3_a, lena256H * 3.25, lena256W * 3.25, 0.4);

			// b)
			unsigned char* img3_b = bicubic(imgLena256, lena256H, lena256W, .4);
			img3_b = bicubic(img3_b, sideLen, sideLen, 3.25);

			// c)
			unsigned char* img3_c = bicubic(imgLena256, lena256H, lena256W, 1.3);

			//// Transform Arrays to cv::Mat
			cv::Mat mat3_a(lena256H * 3.25 * .4, lena256W * 3.25 * .4, CV_8UC1, img3_a);
			cv::Mat mat3_b(sideLen * 3.25, sideLen * 3.25, CV_8UC1, img3_b);
			cv::Mat mat3_c(lena256H * 1.3, lena256W * 1.3, CV_8UC1, img3_c);

			// Save Image
			cv::imwrite("1_4_BCa.png", mat3_a);
			cv::imwrite("1_4_BCb.png", mat3_b);
			cv::imwrite("1_4_BCc.png", mat3_c);
			cout << "Image(s) has been saved!" << endl;

			// Show Images
			cv::imshow("1_4: Bicubic ^3.25v2.5", mat3_a);
			cv::imshow("1_4: bicubic v2.5^3.25", mat3_b);
			cv::imshow("1_4: bicubic ^ 1.3", mat3_c);
			cv::waitKey(0);
			cv::destroyAllWindows();

			delete[] img3_a, delete[] img3_b, delete[] img3_c;
		}

		// 2_1 MSE and PSNR Score between
		else if (select == 5) {
			unsigned char* noisyImg = rcReplicate(imgLena256, lena256H, lena256W, 2, 2);
			psnr(imgLena512, noisyImg, 512, 512, true);
			delete[] noisyImg;
		}
		
		// 2_2 Resize lena_256.raw with v2^2 and v4^4 and compute MSE and PSNR
		else if (select == 6) {
			// 1) v2^2
			unsigned char* resizedImg1 = rcDeletion(imgLena256, lena256H, lena256W, 2, 2);
			resizedImg1 = rcReplicate(resizedImg1, lena256H / 2, lena256W / 2, 2, 2);
			cout << "For Question 2_2_1: " << endl
				 << "-----------------------------------" << endl;
			psnr(imgLena256, resizedImg1, lena256H, lena256W);
			cout << endl;

			cv::Mat matOriginal(lena256H, lena256W, CV_8UC1, imgLena256);
			cv::Mat matResized(lena256H, lena256W, CV_8UC1, resizedImg1);
			cv::imshow("Original Image", matOriginal);
			cv::imshow("Resized Image", matResized);
			cv::waitKey(0);
			cv::destroyAllWindows();

			delete resizedImg1;

			// 2) v4^4
			unsigned char* resizedImg2 = rcDeletion(imgLena256, lena256H, lena256W, 4, 4);
			resizedImg2 = rcReplicate(resizedImg2, lena256H / 4, lena256W / 4, 4, 4);
			cout << "For Question 2_2_2: " << endl
				 << "-----------------------------------" << endl;
			psnr(imgLena256, resizedImg2, lena256H, lena256W);
			cout << endl;

			cv::Mat matResized2(lena256H, lena256W, CV_8UC1, resizedImg2);
			cv::imshow("Original Image", matOriginal);
			cv::imshow("Resized Image", matResized2);
			cv::waitKey(0);
			cv::destroyAllWindows();

			delete resizedImg2;
		}

		// 3_1 Reduce Spatial Resolution
		else if (select == 7) {
			float scale = 0.5;
			unsigned char* damagedImg = bicubic(imgLena512, 512, 512, scale);
			damagedImg = bicubic(damagedImg, 512 * scale, 512 * scale, 1 / scale);
			psnr(imgLena512, damagedImg, 512, 512);

			// Transform Arrays to cv::Mat
			cv::Mat matDamaged(512, 512, CV_8UC1, damagedImg);

			// Save Image
			cv::imwrite("3_1.png", matDamaged);
			cout << "Image(s) has been saved!" << endl;

			// Show Images
			cv::imshow("3_1: Received Image", matDamaged);
			cv::waitKey(0);
			cv::destroyAllWindows();

			delete[] damagedImg;
		}
		
		// 3_2 Reduce Color Resolution
		else if (select == 8){
			int desBit = 4;
			unsigned char* quantizedImg = colorQuantization(imgLena512, 512, 512, desBit);
			psnr(quantizedImg, imgLena512, 512, 512);
			cout << "Image has been transformed to " << desBit << "bits." << endl
				 << "Thus the new image size is 512 * 512 * 4 / 8 = " << 512 * 512 * 4 / 8 << " Bytes" << endl << endl;

			// Transform Arrays to cv::Mat
			cv::Mat matQuantized(512, 512, CV_8UC1, quantizedImg);

			// Save Image
			cv::imwrite("3_2.png", matQuantized);
			cout << "Image(s) has been saved!" << endl;

			// Show Images
			cv::imshow("3_2: Received Image", matQuantized);
			cv::waitKey(0);
			cv::destroyAllWindows();

			delete[] quantizedImg;
		}

		// 3_3 Mixing 3_1 and 3_2 to find best balance
		else if (select == 9) {
			// Test Phase
			float scale = .5;

			unsigned char* resizedImg = bicubic(imgLena512, 512, 512, scale);
			resizedImg = bicubic(resizedImg, 512 * scale, 512 * scale, 1 / scale);
			cout << endl << "Reduce image size by reducing width and height to half: " << endl;
			psnr(imgLena512, resizedImg, 512, 512);

			int sendBit = 2;
			unsigned char* quantizedImg = colorQuantization(imgLena512, 512, 512, sendBit);
			cout << endl << "Reduce image size by reducing bits to 2: " << endl;
			psnr(imgLena512, quantizedImg, 512, 512);

			// Practical Phase
			scale = .75;
			int desBit = 7;
			unsigned char* practicalImg = bicubic(imgLena512, 512, 512, scale);
			practicalImg = colorQuantization(practicalImg, 384, 384, desBit);
			practicalImg = bicubic(practicalImg, 512 * scale, 512 * scale, 1 / scale);
			cout << endl << "Reduce image by mixing 2 methods: " << endl;
			psnr(imgLena512, practicalImg, 512, 512);

			// Transform Arrays to cv::Mat
			cv::Mat matCompressed(512, 512, CV_8UC1, practicalImg);

			// Save Image
			cv::imwrite("3_3.png", matCompressed);
			cout << "Image(s) has been saved!" << endl;

			// Show Images
			cv::imshow("3_3: Received Image", matCompressed);
			cv::waitKey(0);
			cv::destroyAllWindows();
			
			delete[] resizedImg;
			delete[] quantizedImg;
			delete[] practicalImg;
		}

		else { cout << "Try Again! It's not that Hard!" << endl; }
	}

	//---------- Release Memory ----------//
	std::fclose(lena256File);
	std::fclose(lena512File);
	std::fclose(lena512BlurredFile);

	return 0;
}
