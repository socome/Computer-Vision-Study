/*
#include "dirent.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace ml;

Mat train_descriptor(Mat img1);
static bool writeVocabulary(const string& filename, const Mat& vocabulary);
static bool readVocabulary(const string& filename, Mat& vocabulary);

int main()
{

	Ptr<FeatureDetector> Detector = SIFT::create();
	Ptr<DescriptorExtractor> extractor = SIFT::create();

	String folderpath = "D:/Computer-Vision-Study/week2_1/week2_1/dataset/last_test/train/";
	vector<String> filenames;
	Mat training_descriptors_S(1, extractor->descriptorSize(), extractor->descriptorType());
	glob(folderpath, filenames);


	cout << "\n------- file load ---------\n" << endl;

	cout << "Train data file size =  " << filenames.size() << endl;

	for (size_t i = 0; i < filenames.size(); i++)
	{
		vector< KeyPoint > Keypoint_img;
		Mat img1_descriptors;

		Mat img_read = imread(filenames[i], IMREAD_GRAYSCALE);

		Detector->detect(img_read, Keypoint_img);
		extractor->compute(img_read, Keypoint_img, img1_descriptors);

		Mat descriptors = train_descriptor(img_read);
		training_descriptors_S.push_back(descriptors);

		cout << filenames[i] << "  load" << endl;
	}

	Mat vocabulary;
	vocabulary.create(0, 1, CV_32F);
	TermCriteria terminate_criterion;
	terminate_criterion.epsilon = FLT_EPSILON;
	int num_cluster = 100;
	BOWKMeansTrainer bowtrainer(num_cluster, terminate_criterion, 3, KMEANS_PP_CENTERS);
	bowtrainer.add(training_descriptors_S);

	cout << "Total descriptors: " << training_descriptors_S.rows << endl;

	cout << "-----------------\nTrain set\n-----------------\n" << endl;


	vocabulary = bowtrainer.cluster();

	Ptr<DescriptorMatcher> matcher = BFMatcher::create();
	BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	writeVocabulary("vocabulary", vocabulary);

	//cout << "-------vocabulary ---------\n";
	//cout << vocabulary << endl;
	//cout << "\n\n";

	Mat train_samples;
	Mat labels;
	train_samples.create(0, 0, CV_32F);
	labels.create(0, 1, CV_32SC1);

	
	cout << "\n------- positive images ---------\n" << endl;
	
	Mat samples;

	String folderpath_p = "D:/Computer-Vision-Study/week2_1/week2_1/dataset/last_test/pos/";
	vector<String> filenames_p;
	glob(folderpath_p, filenames_p);
	for (size_t i = 0; i < filenames_p.size(); i++)
	{
		vector< KeyPoint > Keypoint_img_p;
		Mat img= imread(filenames_p[i], IMREAD_GRAYSCALE);

		extractor->detect(img, Keypoint_img_p);
		if (Keypoint_img_p.empty()) cout << "No keypoints found." << endl;

		// Responses to the vocabulary
		Mat response_hist_p;
		bowide.compute(img, Keypoint_img_p, response_hist_p);
		if (response_hist_p.empty()) cout << "No descriptors found." << endl;

		cout << "-------imgDescriptor : " << filenames_p[i] << "---------\n";
		cout << response_hist_p << endl;
		cout << "\n";


		if (samples.empty())
		{
			samples.create(0, response_hist_p.cols, response_hist_p.type());
		}

		//Copy class samples and labels
		samples.push_back(response_hist_p);

		Mat classLabels = Mat::ones(response_hist_p.rows, 1, CV_32SC1);
		labels.push_back(classLabels);

	}
	cout << "Adding " << filenames_p.size() << " positive sample." << endl;

	cout << "------- negative images---------" << endl;

	String folderpath_n = "D:/Computer-Vision-Study/week2_1/week2_1/dataset/last_test/neg/";
	vector<String> filenames_n;
	glob(folderpath_n, filenames_n);
	for (size_t i = 0; i < filenames_n.size(); i++)
	{
		vector< KeyPoint > Keypoint_img_n;
		Mat img = imread(filenames_n[i], IMREAD_GRAYSCALE);
		extractor->detect(img, Keypoint_img_n);
		if (Keypoint_img_n.empty()) cout << "No keypoints found." << endl;

		// Responses to the vocabulary
		Mat response_hist_n;
		bowide.compute(img, Keypoint_img_n, response_hist_n);
		if (response_hist_n.empty()) cout << "No descriptors found." << endl;

		cout << "-------imgDescriptor : " << filenames_n[i] << "---------\n";
		cout << response_hist_n << endl;
		cout << "\n\n";

		//Copy class samples and labels
		samples.push_back(response_hist_n);

		Mat classLabels = -Mat::ones(response_hist_n.rows, 1, CV_32SC1);
		labels.push_back(classLabels);


	}
	cout << "Adding " << filenames_n.size() << " negative sample." << endl;


	cout << "\n------- training ---------\n" << endl;

	

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));


	cout << "-------train_samples ---------\n";
	cout << samples << endl;
	cout << "\n\n";

	cout << "-------labels : ---------\n";
	cout << labels << endl;
	cout << "\n\n";


	svm->train(samples, ROW_SAMPLE, labels);
	// Do something with the classifier, like saving it to file

	cout << "-------SVM_train save ---------\n";
	cout << "\n\n";

	svm->save("train.xml");


	cout << "\n------- test ---------\n" << endl;

	String folderpath_test = "D:/Computer-Vision-Study/week2_1/week2_1/dataset/last_test/test/";
	vector<String> filenames_test;
	glob(folderpath_test, filenames_test);
	for (size_t i = 0; i < filenames_test.size(); i++)
	{

		vector< KeyPoint > Keypoint_img_t;

		Mat img_t = imread(filenames_test[i], IMREAD_GRAYSCALE);
		extractor->detect(img_t, Keypoint_img_t);
		if (Keypoint_img_t.empty()) cout << "No keypoints found." << endl;

		// Responses to the vocabulary
		Mat imgDescriptor_t;
		bowide.compute(img_t, Keypoint_img_t, imgDescriptor_t);
		if (imgDescriptor_t.empty()) cout << "No descriptors found." << endl;

		cout << "-------imgDescriptor : " << i << "---------\n";
		cout << imgDescriptor_t << endl;
		cout << "\n\n";

		float reulst = svm->predict(imgDescriptor_t);

		cout << reulst << endl;

		if (reulst == 1) cout << " This is car" << endl;
		else cout << " This is airplane" << endl;

		imshow("test", img_t);
		waitKey(200);

		

	}
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
*/