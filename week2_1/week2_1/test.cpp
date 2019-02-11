
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

void main()
{
	vector< KeyPoint > Keypoint_img_a, Keypoint_img_c;
	Mat img_a_descriptors, img_c_descriptors;

	Ptr<FeatureDetector> Detector = SURF::create();
	Ptr<DescriptorExtractor> Extractor = SURF::create();

	Mat img_a = imread("neg_3");
	Mat img_c = imread("pos_16");

	Detector->detect(img_a, Keypoint_img_a);
	Extractor->compute(img_a, Keypoint_img_a, img_a_descriptors);


	Detector->detect(img_c, Keypoint_img_c);
	Extractor->compute(img_c, Keypoint_img_c, img_c_descriptors);

	Mat training_descriptors_S(1, Extractor->descriptorSize(), Extractor->descriptorType());

	training_descriptors_S.push_back(img_a_descriptors);
	training_descriptors_S.push_back(img_c_descriptors);


	cout << "-------vocabulary ---------\n";
	cout << training_descriptors_S << endl;
	cout << "\n\n";

	
	int num_cluster = 200;

	Mat vocabulary;
	vocabulary.create(0, 1, CV_32FC1);
	BOWKMeansTrainer bowtrainer(num_cluster);
	bowtrainer.add(training_descriptors_S);
	vocabulary = bowtrainer.cluster();

	Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_L2);
	BOWImgDescriptorExtractor bowide(Extractor, matcher);
	bowide.setVocabulary(vocabulary);

	cout << "-------vocabulary ---------\n";
	cout << vocabulary << endl;
	cout << "\n\n";
	
}
*/