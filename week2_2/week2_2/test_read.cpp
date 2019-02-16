/*
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

void main()
{
	ifstream train_data("image_train_label_rs.csv");
	vector<vector<double> > train_data_vector;

	if (!train_data.is_open())
	{
		cout << "Error: File Opne" << endl;
	}
	
	if (train_data)
	{
		int cnt = 0;
		string line;
		while (getline(train_data, line)) 
		{
			vector<double> train_data_vector_2;
			stringstream linStream(line);
			string cell;

			while (getline(linStream, cell, ','))
			{
				train_data_vector_2.push_back(stod(cell));
			}
			train_data_vector.push_back(train_data_vector_2);
		}
	}

	Mat trainLabelsMat(train_data_vector.size(), 1, CV_32FC1);

	for (int i = 0; i < train_data_vector.size(); i++)
	{
		trainLabelsMat.at<float>(i, 0) = train_data_vector[i][0];
	}

}
*/