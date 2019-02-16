/*
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
	vector<vector<float>> CNN_train_data_label = CsvtoVector("image_train_label_rs.csv");
	vector<vector<float>> CNN_train_data = CsvtoVector("image_train_rs.csv");

	int descriptor_size = CNN_train_data[0].size();

	Mat MNIST_CNN_train_data_Mat(CNN_train_data.size(), descriptor_size, CV_32FC1);
	Mat MNIST_CNN_train_data_label_Mat(CNN_train_data_label.size(), 1, CV_32FC1);

	ConvertVectortoMatrix(CNN_train_data, MNIST_CNN_train_data_Mat);

	for (int i = 0; i < CNN_train_data_label.size(); i++)
	{
		MNIST_CNN_train_data_label_Mat.at<float>(i, 0) = CNN_train_data_label[i][0];
	}




	cout << "\n//////////////////////////////////////////////////////" << endl;

	cout << "                      svm setting                      " << endl;

	cout << "\n//////////////////////////////////////////////////////\n" << endl;

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.gamma = 0.50625;
	params.C = 2.5;
	CvSVM svm;

	cout << "\n//////////////////////////////////////////////////////" << endl;

	cout << "                      svm train                      " << endl;

	cout << "\n//////////////////////////////////////////////////////\n" << endl;

	CvMat tryMat = MNIST_CNN_train_data_Mat;
	CvMat tryMat_2 = MNIST_CNN_train_data_label_Mat;
	svm.train(&tryMat, &tryMat_2, Mat(), Mat(), params);

	cout << "\n//////////////////////////////////////////////////////" << endl;

	cout << "                      svm save                   " << endl;

	cout << "\n//////////////////////////////////////////////////////\n" << endl;
	/// Save svm file
	svm.save("MNIST_CNN_SVM.xml");

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
	cout << "finished loading " << endl<<endl;

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
*/