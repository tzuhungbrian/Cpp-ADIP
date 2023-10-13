#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <vector>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>

void printCoordinateElem(unsigned char* imgArr, int row, int col)
{
	int index = row * 256 + col;
	std::cout << "The element at (" << row << "," << col << ") is " << int(imgArr[index]) << std::endl;
}

void printIndexElem(unsigned char* imgArr, int index)
{
	std::cout << "The " << index << "th element is " << int(imgArr[index]) << std::endl;
}

unsigned char* splitBlock(unsigned char* imgArr, int height, int width, int index)
{
	int imgSize = height * width;
	int const subImgSize = 128 * 128;
	unsigned char* upLeftImgArr = new unsigned char[subImgSize];
	unsigned char* lowRightImgArr = new unsigned char[subImgSize];

	int widthCount = 0, heightCount = 0;
	int upLeftCount = 0, lowRightCount = 0;
	for (auto i = 0; i != imgSize; ++i)
	{
		if ((widthCount <= 127) && (heightCount <= 127)) // Upper left image
		{
			upLeftImgArr[upLeftCount] = imgArr[i];
			upLeftCount += 1;
		}
		else if ((widthCount >= 128) && (heightCount >= 128)) // Lower left image
		{
			lowRightImgArr[lowRightCount] = imgArr[i];
			lowRightCount += 1;
		}
		widthCount += 1;
		if (widthCount == width)
		{
			widthCount = 0;
			heightCount += 1;
		}
	}
	if (index == 0)
		return upLeftImgArr;
	else if (index == 3)
		return lowRightImgArr;
	else { std::cout << "Nice try dick!" << std::endl; }
}

unsigned char* rotate90(unsigned char* imgArr, int height, int width)
{
	int size = height * width;
	unsigned char* rotated90Arr = new unsigned char[size];

	int wCount = 0, hCount = 0;
	for (auto i = 0; i != size; ++i)
	{
		int oldIndex = hCount * 128 + wCount;
		int newIndex = wCount * 128 + width - 1 - hCount;

		rotated90Arr[newIndex] = imgArr[i];

		wCount += 1;
		if (wCount == width)
		{
			wCount = 0;
			hCount += 1;
		}
	}
	return rotated90Arr;
}

unsigned char* concatenate(unsigned char* originImg, unsigned char* img0,
	unsigned char* img3, int height, int width)
{	
	int fullSize = height * width;
	unsigned char* fullImgArr = new unsigned char[fullSize];

	int wCount = 0, hCount = 0;
	int img0Count = 0, img3Count = 0;
	for (auto i = 0; i != fullSize; ++i)
	{
		if ((wCount <= 127 && hCount <= 127))
		{
			fullImgArr[i] = img0[img0Count];
			img0Count += 1;
		}
		else if ((wCount >= 128) && (hCount >= 128))
		{
			fullImgArr[i] = img3[img3Count];
			img3Count += 1;
		}
		else
		{
			fullImgArr[i] = originImg[i];
		}
		wCount += 1;
		if (wCount == width)
		{
			wCount = 0;
			hCount += 1;
		}
	}
	return fullImgArr;
}

unsigned char* mosaic(unsigned char* imgArr, int height, int width)
{
	std::vector<int> intensity;
	
	// create vector of sampled points
	for (auto j = 0; j != 32; ++j) // sampling
	{
		for (auto i = j * 2048; i != 2048 * j + width; ++i)
		{
			if (i == 0 || i % 8 == 0) { intensity.push_back(imgArr[i]); }
		}
	}


	int size = height * width;
	unsigned char* newImgArr = new unsigned char[size];

	for (auto arrIndex = 0; arrIndex != 1024; ++arrIndex)
	{
		int x = arrIndex / 32;
		int y = arrIndex % 32;
		for (auto blockRows = x * 8; blockRows != x * 8 + 8; ++blockRows)
		{
			for (auto blockCols = y * 8; blockCols != y * 8 + 8; ++blockCols)
			{
				int index = blockRows * 256 + blockCols;
				newImgArr[index] = intensity[arrIndex];
			}
		}
	}

	return newImgArr;
}

unsigned char* mirrorRight(unsigned char* imgArr, int height, int width)
{
	int newSize = height * width * 2;
	int newWidth = width * 2;
	unsigned char* newArr = new unsigned char[newSize];

	for (auto row = 0; row != height; ++row)
	{
		for (auto col = 0; col != newWidth; ++col) 
		{
			int idx = newWidth * row + col;
			int mirroredIdx = width * row - col + newWidth - 1;
			int originalIdx = width * row + col;
			if (col >= width) { newArr[idx] = imgArr[mirroredIdx]; }
			else { newArr[idx] = imgArr[originalIdx]; }
		}
	}
	return newArr;
}

unsigned char* mirrorDown(unsigned char* imgArr, int height, int width)
{
	int newHeight = height * 2;
	int newSize = newHeight * width;
	unsigned char* newArr = new unsigned char[newSize];

	for (auto row = 0; row != newHeight; ++row) 
	{
		for (auto col = 0; col != width; ++col) 
		{
			int idx = width * row + col;
			int originalIdx = width * (-row + newHeight - 1) + col;
			if (row >= height) { newArr[idx] = imgArr[originalIdx]; }
			else { newArr[idx] = imgArr[idx]; }
		}
	}
	return newArr;
}

unsigned char* increaseIntensity(unsigned char* imgArr, int height, int width,
								int addValue, bool random=false)
{
	int size = height * width;
	unsigned char* newArr = new unsigned char[size];
	for (auto row = 0; row != height; ++row)
	{
		for (auto col = 0; col != width; ++col)
		{
			int idx = row * width + col;
			if (random == true) { addValue = rand() % 101 - 50; }
			else { addValue = addValue; }
			newArr[idx] = imgArr[idx] + addValue;
		}
	}
	return newArr;
}

cv::Mat drawAndPlot(unsigned char* imgArr, int height, int width, bool show = true)
{
	cv::Mat matImg(height, width, CV_8UC1, imgArr);

	// Plotting box
	cv::Point topLeft(460, 570); //(x, y)
	cv::Point botRight(530, 700);
	cv::rectangle(matImg, topLeft, botRight,
		cv::Scalar(0, 0, 0),
		2, cv::LINE_8);

	// Putting Text

	cv::putText(matImg,
		"110318094",
		cv::Point(250, 330), // Coordinates (Bottom-left corner of the text string in the image)
		cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
		2.0, // Scale. 2.0 = 2x bigger
		cv::Scalar(0, 0, 0), // BGR Color
		1, // Line Thickness (Optional)
		cv::LINE_AA); // Anti-alias (Optional, see version note)

	if (show == true) 
	{
		cv::imshow("2", matImg);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
	cv::imwrite("2.png", matImg);
	return matImg;
}

void showImgCV(unsigned char* imgArr, int height, int width, std::string figName)
{
	cv::Mat matrix(height, width, CV_8UC1, imgArr);
	cv::imshow(figName, matrix);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

// The functions below are only used for verifying my homework
// My homework is finished WITHOUT any of below function.
int getIntensityCV(cv::Mat matrix, int row, int col)
{
	uchar val;

	val = matrix.at<uchar>(row, col);
	return int(val);
}

void printIndexElemCV(cv::Mat matrix, int index)
{
	int rows = matrix.rows;
	int cols = matrix.cols;
	int count = -1;

	for (auto i = 0; i != rows; ++i)
	{
		for (auto j = 0; j != cols; ++j)
		{
			count += 1;
			if (count == index)
			{
				float intensity = getIntensityCV(matrix, i, j);
				std::cout << "The coordinate of " << index
					<< "th element is : (" << i << "," << j << ")" << std::endl;
				std::cout << "The intensity at (" << i
					<< "," << j << ") is: " << intensity << std::endl;
			}
		}
	}
}


int main()
{
	//-----------------------1. Initial variable-----------------------//
	char input_img[] = "lena256.raw";                 // Input  raw image name
	char output_img[] = "lena256_out.raw";              // Output raw image name
	FILE* input_file;
	FILE* output_file;
	int width = 256;
	int height = 256;
	int size = width * height;
	unsigned char* imgLena = new unsigned char[size]; // array for image data
	
	char scopeImg[] = "kaleidoscope_cut_2.raw";
	FILE* scopeFile;
	unsigned char* imgScope = new unsigned char[size];

	//-----------------------2. Read File-----------------------//
	// using fopen as example, fstream works too
	input_file = fopen(input_img, "rb");

	if (input_file == NULL)
	{
		puts("Input File Does Not Exist!");
		system("PAUSE");
		exit(0);
	}

	fread(imgLena, sizeof(unsigned char), size, input_file);

	//--------- Read kaleidoscope ---------//
	scopeFile = fopen(scopeImg, "rb");

	if (scopeFile == NULL)
	{
		puts("Kaleidoscope File Not Found!");
		system("PAUSE");
		exit(0);
	}

	fread(imgScope, sizeof(unsigned char), size, scopeFile);

	output_file = fopen(output_img, "wb");
	//--------- Selection Menu ---------//
	std::cout
		<< "0: Exit the program." << std::endl
		<< "1: 1.2.b.1" << std::endl
		<< "2: 1.2.b.2" << std::endl
		<< "3: 1.2.c" << std::endl
		<< "4: 1.2.d" << std::endl
		<< "5: 1.2.e" << std::endl
		<< "6: 1.2.f" << std::endl
		<< "7: 1.3.a" << std::endl
		<< "8: 1.3.b" << std::endl
		<< "9: 2" << std::endl;

	while (true)
	{
		std::cout << "Enter Numbers to Select Answer:" << std::endl;
		int selection = 0;
		std::cin >> selection;

		//--------- Homework ---------//
		if (selection == 0) { break; }
		// 1.2.b.1
		else if (selection == 1) { printCoordinateElem(imgLena, 123, 234); }

		// 1.2.b.2
		else if (selection == 2) { printIndexElem(imgLena, 5487); }

		// 1.2.c
		else if (selection == 3)
		{
			output_file = fopen(output_img, "wb");
			fwrite(imgLena, 1, size, output_file); // Save image as raw
			std::cout << "lena256_out.raw has been saved." << std::endl;
		}

		// 1.2.d
		else if (selection == 4)
		{
			// 1) split
			unsigned char* upLeftBlock = splitBlock(imgLena, 256, 256, 0);
			unsigned char* lowRightBlock = splitBlock(imgLena, 256, 256, 3);
			// 2) manipulate each block
			for (auto i = 0; i != 3; ++i) // rotate 3 times (-90=270=90*3)
			{
				upLeftBlock = rotate90(upLeftBlock, 128, 128);
			}
			lowRightBlock = rotate90(lowRightBlock, 128, 128);
			// 3) concatenate
			unsigned char* fullImg = concatenate(imgLena, upLeftBlock, lowRightBlock, 256, 256);

			char splitAndConcat[] = "1_2_d.raw";
			FILE* splitAndConcatFile = fopen(splitAndConcat, "wb");
			fwrite(fullImg, 1, size, splitAndConcatFile);
			std::cout << "The answer is saved as 1_2_d.raw!" << std::endl;
			showImgCV(fullImg, 256, 256, "1_2_d");

			delete[] upLeftBlock;
			delete[] lowRightBlock;
			delete[] fullImg;

			fclose(splitAndConcatFile);
		}

		// 1.2.e
		else if (selection == 5)
		{
			unsigned char* mosaicImg = mosaic(imgLena, 256, 256);

			char mosaic[] = "1_2_e.raw";
			FILE* mosaicFile = fopen(mosaic, "wb");
			fwrite(mosaicImg, 1, size, mosaicFile);
			std::cout << "The answer is saved as 1_2_e.raw!" << std::endl;

			showImgCV(mosaicImg, 256, 256, "1_2_e");
			delete[] mosaicImg;
			fclose(mosaicFile);
		}

		// 1.2.f
		else if (selection == 6)
		{
			// 1) mirror right * 2
			unsigned char* mirroredScope = mirrorRight(imgScope, 256, 256);
			mirroredScope = mirrorRight(mirroredScope, 256, 512);
			// 2) mirror down * 2
			mirroredScope = mirrorDown(mirroredScope, 256, 1024);
			mirroredScope = mirrorDown(mirroredScope, 512, 1024);

			char mirror[] = "1_2_f.raw";
			FILE* mirrorFile = fopen(mirror, "wb");
			fwrite(mirroredScope, 1, 1024 * 1024, mirrorFile);
			std::cout << "The answer is saved as 1_2_f.raw!" << std::endl;
			showImgCV(mirroredScope, 1024, 1024, "1_2_f");

			delete[] mirroredScope;
			fclose(mirrorFile);
		}

		// 1.3.a
		else if (selection == 7)
		{
			unsigned char* tunedLena = increaseIntensity(imgLena, 256, 256, 50);
			char tunedLenaName[] = "1_3_a.raw";
			FILE* tunedLenaFile = fopen(tunedLenaName, "wb");
			fwrite(tunedLena, 1, size, tunedLenaFile);

			std::cout << "The answer is saved as 1_3_a.raw!" << std::endl;

			delete[] tunedLena;
			fclose(tunedLenaFile);
		}

		// 1.3.b
		else if (selection == 8)
		{
			char randomLenaName[] = "1_3_b.raw";
			FILE* randomLenaFile = fopen(randomLenaName, "wb");
			unsigned char* randomLena = increaseIntensity(imgLena, 256, 256, 50, true);
			fwrite(randomLena, 1, size, randomLenaFile);

			std::cout << "The answer is saved as 1_3_b.raw!" << std::endl;

			delete[] randomLena;
			fclose(randomLenaFile);
		}

		// 2
		else if (selection == 9)
		{
			// 1) Load willy into cv::Mat
			char willyImg[] = "willy_795x826.raw";
			int willyWidth = 795, willyHeight = 826;
			int willySize = willyWidth * willyHeight;
			unsigned char* imgWilly = new unsigned char[willySize];
			FILE* willyFile = fopen(willyImg, "rb");
			fread(imgWilly, sizeof(unsigned char), willySize, willyFile);

			// 2) Plot box and put ID numbers.
			drawAndPlot(imgWilly, willyHeight, willyWidth, false);
			std::cout << "The answer is saved as 2.png!" << std::endl;

			delete[] imgWilly;
			fclose(willyFile);
		}

		else { std::cout << "Try again!"; }

		// Create window and show image.
		cv::Mat mat_lena(height, width, CV_8UC1, imgLena);
		//cv::imshow("lena in opencv", mat_lena);
		//cv::waitKey(0);
		//cv::destroyAllWindows();


		cv::imwrite("lena256out_opencv.jpg", mat_lena); // Save image by OpenCV

	}
	// Release memory
	delete[] imgLena;

	fclose(input_file);
	fclose(output_file);
	return 0;
}
