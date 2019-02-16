
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

void ConvertVectortoMatrix(vector<vector<float> > &In_vector, Mat &Out_Mat);
vector<vector<float>> CsvtoVector(String filename);

int main()
{
	vector<vector<float>> CNN_test_data = CsvtoVector("image_test_rs.csv");
	vector<vector<float>> CNN_test_data_label = CsvtoVector("image_test_label_rs.csv");

	int descriptor_size = CNN_test_data[0].size();

	Mat MNIST_CNN_test_data_Mat(CNN_test_data.size(), descriptor_size, CV_32FC1);
	Mat MNIST_CNN_test_data_label_Mat(CNN_test_data_label.size(), 1, CV_32FC1);

	ConvertVectortoMatrix(CNN_test_data, MNIST_CNN_test_data_Mat);

	for (int i = 0; i < CNN_test_data_label.size(); i++)
	{
		MNIST_CNN_test_data_label_Mat.at<float>(i, 0) = CNN_test_data_label[i][0];
	}

	cout << "\n//////////////////////////////////////////////////////" << endl;

	cout << "                      svm setting                      " << endl;

	cout << "\n//////////////////////////////////////////////////////\n" << endl;

	Mat testResponse;
	CvSVM svm;
	svm.load("./MNIST_CNN_SVM.XML");



	cout << "\n//////////////////////////////////////////////////////" << endl;

	cout << "                      svm predict                      " << endl;

	cout << "\n//////////////////////////////////////////////////////\n" << endl;

	svm.predict(MNIST_CNN_test_data_Mat, testResponse);

	float count = 0, accuracy = 0;

	for (int i = 0; i < testResponse.rows; i++)
	{
		cout << testResponse.at<float>(i, 0) << "   " << CNN_test_data_label[i][0] << endl;

		if (testResponse.at<float>(i, 0) == CNN_test_data_label[i][0])
		{
			count = count + 1;
		}
	}

	accuracy = (count / testResponse.rows) * 100;


	cout << "accuracy : " << accuracy << endl;

	return 0;

}


vector<vector<float>> CsvtoVector(String filename)
{
	ifstream train_data(filename);
	vector<vector<float> > vector_data;

	if (!train_data.is_open())
	{
		cout << "Error: File Opne" << endl;
	}
	else
	{
		cout << "Find : " << filename << endl;
		cout << "starting loading" << endl;
	}

	if (train_data)
	{
		int cnt = 0;
		string line;
		while (getline(train_data, line))
		{
			vector<float> train_data_vector_2;
			stringstream linStream(line);
			string cell;
			cnt++;
			while (getline(linStream, cell, ','))
			{
				train_data_vector_2.push_back(stof(cell));
			}

			vector_data.push_back(train_data_vector_2);
		}
	}
	cout << "finished loading " << endl << endl;

	return vector_data;
}

void ConvertVectortoMatrix(vector<vector<float> > &In_vector, Mat &Out_Mat)
{

	int descriptor_size = In_vector[0].size();

	for (int i = 0; i < In_vector.size(); i++)
	{
		for (int j = 0; j < descriptor_size; j++)
		{
			Out_Mat.at<float>(i, j) = In_vector[i][j];
		}
	}

}
