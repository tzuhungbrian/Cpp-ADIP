#define _CRT_SECURE_NO_DEPRECATE

#include "Image.h"
#include "Utils.h"

using namespace std;

int main()
{
	Image appleNoise("apple_noise_512.raw", 512, 512);
	Image street("night_street_512.raw", 512, 512);
	Image streetBlur("street_blur_512x640.raw", 640, 512);

	//appleNoise.show("apple");
	//street.show("street");
	//streetBlur.show("street blurred");

	popOutMenu();
	while (true) 
	{
		cout << endl << "Enter the Question Number to Show Answer:" << endl;
		int select = 0;
		cin >> select;

		if (select == 0) { break; }

		else if (select == 1) {
			cv::Mat street1 = street.homomorphicFiltering(2, 0.51, 5, 10); // rH larger -> darker
			cv::Mat street2 = street.homomorphicFiltering(0.51, 1.0, 50, 10); // 
			cv::Mat street3 = street.homomorphicFiltering(0.25, 0.75, 5, 10); // rL larger -> dark smirch

			street1.convertTo(street1, CV_8UC1, 255.0);
			street2.convertTo(street2, CV_8UC1, 255.0);
			street3.convertTo(street3, CV_8UC1, 255.0);

			cv::imshow("set 1", street1);
			cv::imshow("set 2", street2);
			cv::imshow("rH = 0.25, rL = 0.75, D0 = 5, c = 10", street3);

			cv::imwrite("1_night_street.png", street3);
			cv::imwrite("1_night_street_1.png", street1);
			cv::imwrite("1_night_street_2.png", street2);

			cv::waitKey(0);
			cv::destroyAllWindows();
		}

		else if (select == 2) {
			int d0 = 15;
			int x = 205;
			int y = 232;
			vector<int> p1{x, y};
			vector<int> p2{ appleNoise.width - x, appleNoise.height - y };

			cv::Mat appleNotched = appleNoise.notchFiltering(d0, p1, p2);	

			appleNotched.convertTo(appleNotched, CV_8UC1, 255.0);
			cv::imshow("Notched", appleNotched);
			cv::imwrite("2_1_apple_notch.png", appleNotched);

			cv::waitKey(0);
			cv::destroyAllWindows();
		}

		else if (select == 3) {
			int outerRadius = 65;
			int innerRadius = 50;

			cv::Mat appleBandReject = appleNoise.bandReject(outerRadius, innerRadius);

			appleBandReject.convertTo(appleBandReject, CV_8UC1, 255.0);
			cv::imshow("Band Reject Filtering", appleBandReject);
			cv::imwrite("2_2_apple_band_reject.png", appleBandReject);

			cv::waitKey(0);
			cv::destroyAllWindows();
		}

		else if (select == 4) {
			//streetBlur.show("Original");
			//cv::Mat observe = checkSpectrum(streetBlur.arr, streetBlur.height, streetBlur.width, true);
			int IFRadius = 45;
			float threshold = 0.1;
			cv::Mat inverseFilter = streetBlur.inverseFiltering(500, threshold);
			cv::Mat inverseFilterRad = streetBlur.inverseFiltering(IFRadius, threshold);

			auto k = 0.00005;
			cv::Mat wienerF = streetBlur.wienerFiltering(k);

			inverseFilter.convertTo(inverseFilter, CV_8UC1, 255.0);
			inverseFilterRad.convertTo(inverseFilterRad, CV_8UC1, 255.0);
			wienerF.convertTo(wienerF, CV_8UC1, 255.0);

			cv::imshow("Inverse Filter", inverseFilter);
			cv::imshow("Inverse Filter, r = 45", inverseFilterRad);
			cv::imshow("Wiener Filter, k = 0.00005", wienerF);

			cv::imwrite("3_inverse_no_rad.png", inverseFilter);
			cv::imwrite("3_inverse_w_rad.png", inverseFilterRad);
			cv::imwrite("3_wiener.png", wienerF);

			cv::waitKey(0);
			cv::destroyAllWindows();

		}
	}

	return 0;
}

