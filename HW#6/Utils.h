#pragma once
#include <string>
using namespace std;

void popOutMenu();
void saveRawAndPng(const char* saveName, int height, int width, unsigned char* saveImgArr);
void showImgCV(unsigned char* imgArr, int height, int width, std::string figName);
std::string typeOfMat(int type);

float mse(unsigned char* aImg, unsigned char* bImg, int height, int width, bool print = true);
vector<vector<float>> genGaussianKernel(int maskSize, float sigma = 1.0, float mean = 0.0);
unsigned char* gaussianBlur(unsigned char* inputImg, int height, int width, int gaussianSize, float sigma = 1.0, float mean = 0.0);
cv::Mat checkSpectrum(uchar* input, int height, int width, bool show);
