#include <stdio.h>
#include <iostream>
#include "utils.h"

using namespace std;
using namespace misc;

void misc::popOutMenu()
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
