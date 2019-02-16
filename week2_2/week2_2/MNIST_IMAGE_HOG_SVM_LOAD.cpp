/*
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

#include <iostream>


using namespace cv;
using namespace std;


void ConvertVectortoMatrix(vector<vector<float>> &testHOG, Mat &testMat);
vector<float> find_HOG_feature_image(Mat img);


/////////////////////////////////////////////
//*****please setting the number of images*****
#define NUMBER_test 500
/////////////////////////////////////////////


int main()
{
	vector<vector<float>> MNIST_test_HOG;
	vector<int> MNIST_test_label;

	cout << "\n//////////////////////////////////////////////////////" << endl;

	cout << "test Image load" << endl;

	for (int num = 0; num < 10; num++)
	{
		cout << num << " image load" << endl;
		for (int i = 1; i < NUMBER_test; i++)
		{
			Mat MNIST = imread("./MNIST/data " + std::to_string(num) + "_" + std::to_string(i + 1) + ".PNG", COLOR_RGB2GRAY);
			if (!(MNIST.data))
			{
				cout << "image load fail" << endl;
				return 0;
			}
			//imshow("test", MNIST);
			//waitKey(1);
			MNIST_test_HOG.push_back(find_HOG_feature_image(MNIST));
			MNIST_test_label.push_back(num);
		}

	}
	cout << "\n//////////////////////////////////////////////////////" << endl;

	cout << "                      svm start                      " << endl;

	cout << "\n//////////////////////////////////////////////////////\n" << endl;

	int descriptor_size = MNIST_test_HOG[0].size();
	Mat MNIST_test_HOG_Mat(MNIST_test_HOG.size(), descriptor_size, CV_32FC1);
	
	ConvertVectortoMatrix(MNIST_test_HOG, MNIST_test_HOG_Mat);

	Mat testResponse;
	CvSVM svm;
	svm.load("./MNIST_HOG_SVM.XML");



	cout << "\n//////////////////////////////////////////////////////" << endl;

	cout << "                      svm predict                      " << endl;

	cout << "\n//////////////////////////////////////////////////////\n" << endl;

	svm.predict(MNIST_test_HOG_Mat, testResponse);

	float count = 0, accuracy = 0;
	for (int i = 0; i < testResponse.rows; i++)
	{
		if (testResponse.at<float>(i, 0) == MNIST_test_label[i])
		{
			count = count + 1;
		}
	}

	accuracy = (count / testResponse.rows) * 100;


	cout << "accuracy : " << accuracy << endl;

	return 0;
}

vector<float> find_HOG_feature_image(Mat img)
{
	HOGDescriptor IMAGE_HOG
	(
		Size(20, 20), //winSize
		Size(8, 8), //blocksize
		Size(4, 4), //blockStride,
		Size(8, 8), //cellSize,
		9, //nbins,
		1, //derivAper,
		-1, //winSigma,
		0, //histogramNormType,
		0.2, //L2HysThresh,
		0,//gammal correction,
		64//nlevels=64
	);



	vector<float> hog_descriptor;
	IMAGE_HOG.compute(img, hog_descriptor);
	return hog_descriptor;

}

void ConvertVectortoMatrix(vector<vector<float>> &testHOG, Mat &testMat)
{

	int descriptor_size = testHOG[0].size();

	for (int i = 0; i < testHOG.size(); i++)
	{
		for (int j = 0; j < descriptor_size; j++)
		{
			testMat.at<float>(i, j) = testHOG[i][j];
		}
	}
}

*/