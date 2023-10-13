#define _CRT_SECURE_NO_DEPRECATE
#define _USE_MATH_DEFINES

#include <stdio.h>
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <math.h>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#include "Image.h"
#include "Utils.h"

using namespace std;

int main()
{
	Image rect("rect_256.raw", 256, 256);
	Image rectRot("rect_rot_256.raw", 256, 256);
	Image circle("circle_256.raw", 256, 256);
	Image lines("lines_256.raw", 256, 256);
	Image cat("cat_512.raw", 512, 512);
	Image watermark("watermark_512.raw", 512, 512);
	Image pancake("pancake_512.raw", 512, 512);

	popOutMenu();
	while (true)
	{
		bool q_1_1 = false;
		cout << endl << "Enter the Question Number to Show Answer:" << endl;
		int select = 0;
		cin >> select;

		if (select == 0) { break; }

		// 1_1 DFT for 4 images with origin shifting and contrast enhancement
		else if (select == 1){
			cout << "Transforming rect_256.raw" << endl;
			unsigned char* dftRect = rect.dft2d();
			cout << "Transforming rect_rot_256.raw" << endl;
			unsigned char* dftRectRot = rectRot.dft2d();
			cout << "Transforming circle_256.raw" << endl;
			unsigned char* dftCircle = circle.dft2d();
			cout << "Transforming lines_256.raw" << endl;
			unsigned char* dftLines = lines.dft2d();

			saveRawAndPng("1_1_dftRect", rect.height, rect.width, dftRect);
			saveRawAndPng("1_1_dftRectRot", rectRot.height, rectRot.width, dftRectRot);
			saveRawAndPng("1_1_dftCircle", circle.height, circle.width, dftCircle);
			saveRawAndPng("1_1_dftLines", lines.height, lines.width, dftLines);

			cv::Mat matRect(rect.height, rect.width, CV_8UC1, dftRect);
			cv::Mat matRectRot(rectRot.height, rectRot.width, CV_8UC1, dftRectRot);
			cv::Mat matCircle(circle.height, circle.width, CV_8UC1, dftCircle);
			cv::Mat matLines(lines.height, lines.width, CV_8UC1, dftLines);

			cv::imshow("DFT of rect", matRect);
			cv::imshow("DFT of rectRot", matRectRot);
			cv::imshow("DFT of circle", matCircle);
			cv::imshow("DFT of lines", matLines);

			cv::waitKey(0);
			cv::destroyAllWindows();

			q_1_1 = true;
		}

		// 1_2 DFT by OpenCV for 4 images.
		else if (select == 2) {
			bool isIDFT = false;
			cv::Mat matRect = rect.dft2dCV(rect.arr, isIDFT, false);
			cv::Mat matRectRot = rectRot.dft2dCV(rectRot.arr, isIDFT, false);
			cv::Mat matCircle = circle.dft2dCV(circle.arr, isIDFT, false);
			cv::Mat matLines = lines.dft2dCV(lines.arr, isIDFT, false);

			cv::imshow("DFT of rect", matRect);
			cv::imshow("DFT of rectRot", matRectRot);
			cv::imshow("DFT of circle", matCircle);
			cv::imshow("DFT of lines", matLines);

			matRect.convertTo(matRect, CV_8UC1, 255.0);
			matRectRot.convertTo(matRectRot, CV_8UC1, 255.0);
			matCircle.convertTo(matCircle, CV_8UC1, 255.0);
			matLines.convertTo(matLines, CV_8UC1, 255.0);

			cv::imwrite("1_2_dftRect.png", matRect);
			cv::imwrite("1_2_dftRectRot.png", matRectRot);
			cv::imwrite("1_2_dftCircle.png", matCircle);
			cv::imwrite("1_2_dftLines.png", matLines);

			cv::waitKey(0);
			cv::destroyAllWindows();
		}

		// 1_3 IDFT by formula and mse
		else if (select == 3) {
			if (q_1_1 == true) {
				cout << "Transforming rect_256.raw" << endl;
				unsigned char* idftRect = rect.idft2d();
				cout << "Transforming rect_rot_256.raw" << endl;
				unsigned char* idftRectRot = rectRot.idft2d();
				cout << "Transforming circle_256.raw" << endl;
				unsigned char* idftCircle = circle.idft2d();
				cout << "Transforming lines_256.raw" << endl;
				unsigned char* idftLines = lines.idft2d();

				saveRawAndPng("1_3_dftRect", rect.height, rect.width, idftRect);
				saveRawAndPng("1_3_dftRectRot", rectRot.height, rectRot.width, idftRectRot);
				saveRawAndPng("1_3_dftCircle", circle.height, circle.width, idftCircle);
				saveRawAndPng("1_3_dftLines", lines.height, lines.width, idftLines);

				cv::Mat matRect(rect.height, rect.width, CV_8UC1, idftRect);
				cv::Mat matRectRot(rectRot.height, rectRot.width, CV_8UC1, idftRectRot);
				cv::Mat matCircle(circle.height, circle.width, CV_8UC1, idftCircle);
				cv::Mat matLines(lines.height, lines.width, CV_8UC1, idftLines);

				cv::imshow("IDFT of rect", matRect);
				cv::imshow("IDFT of rectRot", matRectRot);
				cv::imshow("IDFT of circle", matCircle);
				cv::imshow("IDFT of lines", matLines);

				cv::waitKey(0);
				cv::destroyAllWindows();

				float mseRect = mse(rect.arr, idftRect, 256, 256, false);
				float mseRectRot = mse(rectRot.arr, idftRectRot, 256, 256, false);
				float mseCircle = mse(circle.arr, idftCircle, 256, 256, false);
				float mseLines = mse(lines.arr, idftLines, 256, 256, false);

				cout << "MSE of rect = " << mseRect << endl;
				cout << "MSE of rect_rot = " << mseRectRot << endl;
				cout << "MSE of circle = " << mseCircle << endl;
				cout << "MSE of lines = " << mseLines << endl;
			}
			else { cout << "Execute q_1_1 first." << endl; }
		}

		// 1_4 IDFT by OpenCV
		else if (select == 4) {
			bool idft = true;
			cv::Mat matRect = rect.dft2dCV(rect.arr, idft, false);
			cv::Mat matRectRot = rectRot.dft2dCV(rectRot.arr, idft, false);
			cv::Mat matCircle = circle.dft2dCV(circle.arr, idft, false);
			cv::Mat matLines = lines.dft2dCV(lines.arr, idft, false);

			cv::imshow("IDFT of rect", matRect);
			cv::imshow("IDFT of rectRot", matRectRot);
			cv::imshow("IDFT of circle", matCircle);
			cv::imshow("IDFT of lines", matLines);

			matRect.convertTo(matRect, CV_8UC1, 255.0);
			matRectRot.convertTo(matRectRot, CV_8UC1, 255.0);
			matCircle.convertTo(matCircle, CV_8UC1, 255.0);
			matLines.convertTo(matLines, CV_8UC1, 255.0);

			cv::imwrite("1_4_idftRect.png", matRect);
			cv::imwrite("1_4_idftRectRot.png", matRectRot);
			cv::imwrite("1_4_idftCircle.png", matCircle);
			cv::imwrite("1_4_idftLines.png", matLines);

			cv::waitKey(0);
			cv::destroyAllWindows();
		}

		// 2_1 Ideal LPF and HPF with D0 = 5, 20, 50
		else if (select == 5) {
			bool highPass = false;	

			cv::Mat catLPFD5 = cat.idealFiltering(5, "2_1_cat_LP_spec_d5", highPass, true);
			cv::Mat catLPFD20 = cat.idealFiltering(20, "2_1_cat_LP_spec_d20", highPass, true);
			cv::Mat catLPFD50 = cat.idealFiltering(50, "2_1_cat_LP_spec_d50", highPass, true);

			catLPFD5.convertTo(catLPFD5, CV_8UC1, 255.0);
			catLPFD20.convertTo(catLPFD20, CV_8UC1, 255.0);
			catLPFD50.convertTo(catLPFD50, CV_8UC1, 255.0);

			cv::imwrite("2_1_cat_LP_d5.png", catLPFD5);
			cv::imwrite("2_1_cat_LP_d20.png", catLPFD20);
			cv::imwrite("2_1_cat_LP_d50.png", catLPFD50);

			cv::imshow("Ideal Low-Pass Filter, d0 = 5", catLPFD5);
			cv::imshow("Ideal Low-Pass Filter, d0 = 20", catLPFD20);
			cv::imshow("Ideal Low-Pass Filter, d0 = 50", catLPFD50);

			cv::waitKey(0);
			cv::destroyAllWindows();

			// High-Pass Filtering
			highPass = true;
			cv::Mat catHPFD5 = cat.idealFiltering(5, "2_1_cat_HP_spec_d5", highPass,true);
			cv::Mat catHPFD20 = cat.idealFiltering(20, "2_1_cat_HP_spec_d20", highPass, true);
			cv::Mat catHPFD50 = cat.idealFiltering(50, "2_1_cat_HP_spec_d50", highPass, true);

			catHPFD5.convertTo(catHPFD5, CV_8UC1, 255.0);
			catHPFD20.convertTo(catHPFD20, CV_8UC1, 255.0);
			catHPFD50.convertTo(catHPFD50, CV_8UC1, 255.0);

			cv::imwrite("2_1_cat_HP_d5.png", catHPFD5);
			cv::imwrite("2_1_cat_HP_d20.png", catHPFD20);
			cv::imwrite("2_1_cat_HP_d50.png", catHPFD50);

			cv::imshow("Ideal High-Pass Filter, d0 = 5", catHPFD5);
			cv::imshow("Ideal High-Pass Filter, d0 = 20", catHPFD20);
			cv::imshow("Ideal High-Pass Filter, d0 = 50", catHPFD50);

			cv::waitKey(0);
			cv::destroyAllWindows();
		}

		// 2_2 Gaussian LPF and HPF with D0 = 5, 20, 50
		else if (select == 6) {
		bool highPass = false;

		cv::Mat catGLPFD5 = cat.gaussianFiltering(5, "2_2_cat_GLP_spec_d5", highPass, true);
		cv::Mat catGLPFD20 = cat.gaussianFiltering(20, "2_2_cat_GLP_spec_d20", highPass, true);
		cv::Mat catGLPFD50 = cat.gaussianFiltering(50, "2_2_cat_GLP_spec_d50", highPass, true);

		catGLPFD5.convertTo(catGLPFD5, CV_8UC1, 255.0);
		catGLPFD20.convertTo(catGLPFD20, CV_8UC1, 255.0);
		catGLPFD50.convertTo(catGLPFD50, CV_8UC1, 255.0);

		cv::imwrite("2_2_cat_GLP_d5.png", catGLPFD5);
		cv::imwrite("2_2_cat_GLP_d20.png", catGLPFD20);
		cv::imwrite("2_2_cat_GLP_d50.png", catGLPFD50);

		cv::imshow("Gaussian Low-Pass Filter, d0 = 5", catGLPFD5);
		cv::imshow("Gaussian Low-Pass Filter, d0 = 20", catGLPFD20);
		cv::imshow("Gaussian Low-Pass Filter, d0 = 50", catGLPFD50);

		cv::waitKey(0);
		cv::destroyAllWindows();

		// High-Pass Filtering
		highPass = true;
		cv::Mat catGHPFD5 = cat.gaussianFiltering(5, "2_2_cat_GHP_spec_d5", highPass, true);
		cv::Mat catGHPFD20 = cat.gaussianFiltering(20, "2_2_cat_GHP_spec_d20", highPass, true);
		cv::Mat catGHPFD50 = cat.gaussianFiltering(50, "2_2_cat_GHP_spec_d50", highPass, true);

		catGHPFD5.convertTo(catGHPFD5, CV_8UC1, 255.0);
		catGHPFD20.convertTo(catGHPFD20, CV_8UC1, 255.0);
		catGHPFD50.convertTo(catGHPFD50, CV_8UC1, 255.0);

		cv::imwrite("2_2_cat_GHP_d5.png", catGHPFD5);
		cv::imwrite("2_2_cat_GHP_d20.png", catGHPFD20);
		cv::imwrite("2_2_cat_GHP_d50.png", catGHPFD50);

		cv::imshow("Gaussian High-Pass Filter, d0 = 5", catGHPFD5);
		cv::imshow("Gaussian High-Pass Filter, d0 = 20", catGHPFD20);
		cv::imshow("Gaussian High-Pass Filter, d0 = 50", catGHPFD50);

		cv::waitKey(0);
		cv::destroyAllWindows();

		}

		// 3_1 Apply Watermark
		else if (select == 7) {
			//watermark.show("Watermark");
			float normalK = 15.0;
			float largeK = 200.0;
			cv::Mat WMPancakeNormalK = pancake.applyWatermark(watermark.arr, normalK, "3_1_watermarked_pancake_spec_normal_k", true);
			cv::Mat WMPancakeLargeK = pancake.applyWatermark(watermark.arr, largeK, "3_1_watermarked_pancake_spec_large_k", true);

			WMPancakeNormalK.convertTo(WMPancakeNormalK, CV_8UC1, 255.0);
			WMPancakeLargeK.convertTo(WMPancakeLargeK, CV_8UC1, 255.0);

			cv::imwrite("3_1_watermarked_pancake_normal_k.png", WMPancakeNormalK);
			cv::imwrite("3_1_watermarked_pancake_large_k.png", WMPancakeLargeK);

			cv::imshow("Watermarked Pancake, k = 15", WMPancakeNormalK);
			cv::imshow("Watermarked Pancake, k = 200", WMPancakeLargeK);
			cv::waitKey(0);
			cv::destroyAllWindows();
		}

		// 3_2 Apply Gaussian on Watermarked Image
		else if (select == 8) {
			float k = 5;
			int gaussianSize = 3; // 3x3
			float sigma = 1.0;

			cv::Mat watermarkedPancakeMat = pancake.applyWatermark(watermark.arr, k, "_", false);
			watermarkedPancakeMat.convertTo(watermarkedPancakeMat, CV_8UC1, 255.0);
			uchar* watermarkedPancakeArr = watermarkedPancakeMat.data;
			uchar* blurredPancake = gaussianBlur(watermarkedPancakeArr, pancake.height, pancake.width, gaussianSize, sigma, 0.0);
			saveRawAndPng("3_2_blurred_watermarked_pancake", pancake.height, pancake.width, blurredPancake);

			cv::Mat blurredPancakeSpec = checkSpectrum(blurredPancake, pancake.height, pancake.width, true);
			blurredPancakeSpec.convertTo(blurredPancakeSpec, CV_8UC1, 255.0);
			cv::imwrite("3_2_blurred_watermarked_pancake_spectrum.png", blurredPancakeSpec);
		}

		else { continue; }
	}

	return 0;
}