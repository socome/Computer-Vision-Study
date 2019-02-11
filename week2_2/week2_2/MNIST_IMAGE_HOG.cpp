#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

#include <iostream>


using namespace cv;
using namespace std;



void ConvertVectortoMatrix(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, Mat &trainMat, Mat &testMat);
vector<float> find_HOG_feature_image(Mat img);

/////////////////////////////////////////////
//*****please setting the number of images*****
#define NUMBER_train 450
#define NUMBER_test 50
/////////////////////////////////////////////

int main()
{

	vector<vector<float>> MNIST_train_HOG, MNIST_test_HOG;
	vector<int> MNIST_train_label, MNIST_test_label;

	cout << "\n//////////////////////////////////////////////////////" << endl;

	cout << "train Image load" << endl;

	for(int num=0; num<10; num++)
	{
		cout << num << " image load" << endl;
		for (int i = 0; i < NUMBER_train; i++)
		{
			Mat MNIST = imread("./MNIST/data "+ std::to_string(num) + "_" +std::to_string(i + 1) + ".PNG",COLOR_RGB2GRAY);
			if (!(MNIST.data))
			{
				cout << "image load fail" << endl;
				return 0;
			}
			//imshow("train", MNIST);
			//waitKey(1);
			MNIST_train_HOG.push_back(find_HOG_feature_image(MNIST));
			MNIST_train_label.push_back(num);
		}

	}
	cout << "\n//////////////////////////////////////////////////////" << endl;

	cout << "test Image load" << endl;

	for (int num = 0; num < 10; num++)
	{
		cout << num << " image load" << endl;
		for (int i = NUMBER_train; i < NUMBER_test+ NUMBER_train; i++)
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


	int descriptor_size = MNIST_train_HOG[0].size();
	Mat MNIST_train_HOG_Mat(MNIST_train_HOG.size(), descriptor_size, CV_32FC1);
	Mat MNIST_test_HOG_Mat(MNIST_test_HOG.size(), descriptor_size, CV_32FC1);
	ConvertVectortoMatrix(MNIST_train_HOG, MNIST_test_HOG, MNIST_train_HOG_Mat, MNIST_test_HOG_Mat);


	cout << "\n//////////////////////////////////////////////////////" << endl;

	cout << "                      svm setting                      " << endl;

	cout << "\n//////////////////////////////////////////////////////\n" << endl;

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.gamma = 0.50625;
	params.C = 2.5;
	CvSVM svm;


	CvMat tryMat = MNIST_train_HOG_Mat;
	Mat trainLabelsMat(MNIST_train_label.size(), 1, CV_32FC1);

	for (int i = 0; i < MNIST_train_label.size(); i++)
	{
		trainLabelsMat.at<float>(i, 0) = MNIST_train_label[i];
	}

	CvMat tryMat_2 = trainLabelsMat;
	svm.train(&tryMat, &tryMat_2, Mat(), Mat(), params);
	Mat testResponse;
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

void ConvertVectortoMatrix(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, Mat &trainMat, Mat &testMat)
{

	int descriptor_size = trainHOG[0].size();

	for (int i = 0; i < trainHOG.size(); i++)
	{
		for (int j = 0; j < descriptor_size; j++) 
		{
			trainMat.at<float>(i, j) = trainHOG[i][j];
		}
	}
	for (int i = 0; i < testHOG.size(); i++)
	{
		for (int j = 0; j < descriptor_size; j++)
		{
			testMat.at<float>(i, j) = testHOG[i][j];
		}
	}
}