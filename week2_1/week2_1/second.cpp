#include "dirent.h"
#include <iostream>
#include <iostream>
#include <io.h>
#include <stdlib.h>
#include <direct.h>
#include "corecrt_io.h"

#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace ml;

#define SMPLE_CLASS 20

Mat train_descriptor(Mat img1);
static bool writeVocabulary(const string& filename, const Mat& vocabulary);
static bool readVocabulary(const string& filename, Mat& vocabulary);

String folderpath = "./101_ObjectCategories/*.*";
String folder_folderpath = "./101_ObjectCategories/";

int main()
{
	Ptr<FeatureDetector> Detector = SIFT::create();
	Ptr<DescriptorExtractor> extractor = SIFT::create();
	/*
	Mat training_descriptors_S;

	struct _finddata_t fd;
	intptr_t handle;
	if ((handle = _findfirst(folderpath.c_str(), &fd)) == -1L) cout << "No file in directory!" << endl;
	int i = 0;
	int sample_cnt = 0;

	do{
		if(i>= 2)
		{
			vector<String> filenames;

			glob(folder_folderpath + fd.name, filenames);

			if (filenames.size() > 100) continue;
			else
			{
				cout << fd.name << endl;
				sample_cnt++;
			}


			for (size_t j = 0; j < filenames.size(); j++)
			{

				cout << ".....";

				vector< KeyPoint > Keypoint_img;
				Mat img1_descriptors;

				Mat img_read = imread(filenames[j], IMREAD_GRAYSCALE);

				Detector->detect(img_read, Keypoint_img);
				extractor->compute(img_read, Keypoint_img, img1_descriptors);

				Mat descriptors = train_descriptor(img_read);
				training_descriptors_S.push_back(descriptors);
				
				cout << "\b\b\b\b\b";
				
			}
		}

		i++;

		if (sample_cnt > SMPLE_CLASS) break;

	} while (_findnext(handle, &fd) == 0 );

	_findclose(handle);

	cout << "-----------------\nBOW START\n-----------------" << endl;

	Mat vocabulary;
	TermCriteria terminate_criterion;
	terminate_criterion.epsilon = FLT_EPSILON;
	int num_cluster = 300;
	BOWKMeansTrainer bowtrainer(num_cluster, terminate_criterion, 3, KMEANS_PP_CENTERS);
	bowtrainer.add(training_descriptors_S);

	cout << "Total descriptors: " << training_descriptors_S.rows << endl;

	cout << "-----------------\nTRAIN SET\n-----------------\n" << endl;


	vocabulary = bowtrainer.cluster();

	Ptr<DescriptorMatcher> matcher = BFMatcher::create();
	BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	writeVocabulary("vocabulary", vocabulary);
	*/

		
	Mat vocabulary;
	readVocabulary("vocabulary", vocabulary);

	Ptr<DescriptorMatcher> matcher = BFMatcher::create();
	BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	Mat train_samples;
	Mat labels;
	
	Mat train_answer = (Mat_<float>(43, 1) << 1, 9, 7, 9, 11, 8, 8, 4, 7, 2, 7, 5, 9, 11, 6, 7, 5, 8, 5, 6, 11, 8, 11, 6, 8, 5, 4, 3, 3, 3, 11, 1, 9, 2, 3, 5, 4, 11, 11, 10, 10, 10, 10);
	Mat test_answer = (Mat_<float>(39, 1) << 3, 7, 7, 9, 10, 11, 8, 11, 10, 9, 3, 9, 10, 8, 7, 11, 6, 8, 4, 6, 5, 5, 1, 2, 1, 4, 5, 10, 6, 2, 3, 9, 1, 4, 11, 5, 1, 8, 8);

	cout << "\n------- SVM TRAIN ---------\n" << endl;
	

	Mat SVM_train_data(0,1000, CV_32FC1);
	Mat SVM_train_label(0,1, CV_32FC1);

	//(CNN_test_data_label.size(), 1, CV_32FC1)

	struct _finddata_t fd_train;
	intptr_t handle_train;
	if ((handle_train = _findfirst(folderpath.c_str(), &fd_train)) == -1L) cout << "No file in directory!" << endl;

	int i = 0;
	int sample_cnt = 0;

	do{
		if(i>= 2)
		{
			vector<String> filenames;

			glob(folder_folderpath + fd_train.name, filenames);

			if (filenames.size() > 100) continue;
			else
			{
				sample_cnt++;
				cout << fd_train.name << " : " << sample_cnt << endl;
			}


			for (size_t j = 0; j < filenames.size(); j++)
			{

				cout << ".....";

				vector< KeyPoint > Keypoint_img_train;

				Mat img_read = imread(filenames[j], IMREAD_GRAYSCALE);

				extractor->detect(img_read, Keypoint_img_train);
				if (Keypoint_img_train.empty()) cout << "No keypoints found." << endl;

				// Responses to the vocabulary
				Mat response_hist_train;

				bowide.compute(img_read, Keypoint_img_train, response_hist_train);
				if (response_hist_train.empty()) cout << "No descriptors found." << endl;

				SVM_train_data.push_back(response_hist_train);
				SVM_train_label.push_back(sample_cnt);

				
				cout << "\b\b\b\b\b";


			}
		}

		i++;

		if (sample_cnt > SMPLE_CLASS) break;

	} while (_findnext(handle_train, &fd_train) == 0 );

	_findclose(handle_train);

	
	cout << "\n------- SVM TRAIN AUTO ---------\n" << endl;


	Ptr<SVM> svm = SVM::create(); 
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setTermCriteria(TermCriteria(TermCriteria::EPS, 100, 1e-6));
	svm->setGamma(1.0);
	svm->setC(1.0);


	// Train the SVM with given parameters
	Ptr<TrainData> td = TrainData::create(SVM_train_data, ROW_SAMPLE, SVM_train_label);
	svm->trainAuto(td);
	svm->save("svm_train.xml");

	cout << "-------SVM_TRAIN SAVE ---------\n";
	cout << "\n\n";

	Mat result_train;
	Mat result_test;

	cout << "\n------- TRAIN ---------\n" << endl;

	Mat TRAIN_descriptor;	// Responses to the vocabulary
	String folderpath_train = "./train_image/";
	vector<String> filenames_train;
	glob(folderpath_train, filenames_train);

	float count_train = 0, accuracy_train = 0;

	for (size_t i = 0; i < filenames_train.size(); i++)
	{

		vector< KeyPoint > Keypoint_img_train;

		Mat img_t = imread(filenames_train[i], IMREAD_GRAYSCALE);
		extractor->detect(img_t, Keypoint_img_train);
		if (Keypoint_img_train.empty()) cout << "No keypoints found." << endl;

		// Responses to the vocabulary
		Mat imgDescriptor_train;
		bowide.compute(img_t, Keypoint_img_train, imgDescriptor_train);
		if (imgDescriptor_train.empty()) cout << "No descriptors found." << endl;

		TRAIN_descriptor.push_back(imgDescriptor_train);

	}

	svm->predict(TRAIN_descriptor, result_train);

	for (i = 0; i < train_answer.rows; i++)
	{
		if (result_train.at<float>(i, 0) == train_answer.at<float>(i, 0))
		{
			count_train = count_train + 1;
		}
	}

	accuracy_train = (count_train / train_answer.rows) * 100;
	cout << "accuracy : " << accuracy_train << endl;



	cout << "\n------- TEST ---------\n" << endl;


	Mat TEST_descriptor;	// Responses to the vocabulary
	String folderpath_test = "./test_image/";
	vector<String> filenames_test;
	glob(folderpath_test, filenames_test);

	float count_test = 0, accuracy_test = 0;

	for (size_t i = 0; i < filenames_test.size(); i++)
	{
		Mat imgDescriptor_test;

		vector< KeyPoint > Keypoint_img_test;

		Mat img_t = imread(filenames_test[i], IMREAD_GRAYSCALE);
		extractor->detect(img_t, Keypoint_img_test);
		if (Keypoint_img_test.empty()) cout << "No keypoints found." << endl;

		bowide.compute(img_t, Keypoint_img_test, imgDescriptor_test);
		if (imgDescriptor_test.empty()) cout << "No descriptors found." << endl;

		TEST_descriptor.push_back(imgDescriptor_test);

	}

	svm->predict(TEST_descriptor, result_test);

	cout << "test : \n" << result_test << endl;

	for (i = 0; i < test_answer.rows; i++)
	{
		if (result_test.at<float>(i, 0) == test_answer.at<float>(i, 0))
		{
			count_test = count_test + 1;
		}
	}


	accuracy_test = (count_test / test_answer.rows) * 100;

	cout << "accuracy : " << accuracy_test << endl;
	
	return 0;
	
	
}

Mat train_descriptor(Mat img)
{
	if (img.empty())
	{
		cout << "image load fail" << std::endl;
	}
	
	vector< KeyPoint > Keypoint_img;
	Mat img1_descriptors;

	Ptr<FeatureDetector> Detector = SIFT::create();
	Ptr<DescriptorExtractor> Extractor = SIFT::create();
	Detector->detect(img, Keypoint_img);
	Extractor->compute(img, Keypoint_img, img1_descriptors);

	return img1_descriptors;
}



static bool writeVocabulary(const string& filename, const Mat& vocabulary)
{
	cout << "Saving vocabulary..." << endl;
	FileStorage fs(filename, FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "vocabulary" << vocabulary;
		return true;
	}
	return false;
}

static bool readVocabulary(const string& filename, Mat& vocabulary)
{
	cout << "Reading vocabulary...";
	FileStorage fs(filename, FileStorage::READ);
	if (fs.isOpened())
	{
		fs["vocabulary"] >> vocabulary;
		cout << "done" << endl;
		return true;
	}
	return false;
}
