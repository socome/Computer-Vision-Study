#include "dirent.h"
#include <iostream>
#include <iostream>
#include <fstream>
#include <sstream> 
#include <string> 
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

#define SMPLE_CLASS 102
#define num_cluster 200
#define SPATIAL_LEVEL 2
#define SELECT_LEVE 2
#define SPM_MODE 2 // (1: NOMAL / 2: PYRAMID)
#define STEP_SIZE 8

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

			cout << fd.name << endl;
			sample_cnt++;


			for (size_t j = 0; j < filenames.size(); j++)
			{

				cout << ".....";

				vector< KeyPoint > Keypoint_img;
				Mat img1_descriptors;

				Mat img_read = imread(filenames[j],IMREAD_GRAYSCALE);

				for (int m = STEP_SIZE; m < img_read.rows - STEP_SIZE; m += STEP_SIZE)
				{
					for (int n = STEP_SIZE; n < img_read.cols - STEP_SIZE; n += STEP_SIZE)
					{
						Keypoint_img.push_back(KeyPoint(float(n), float(m), float(STEP_SIZE)));
					}
				}
				extractor->compute(img_read, Keypoint_img, img1_descriptors);

				training_descriptors_S.push_back(img1_descriptors);

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
	BOWKMeansTrainer bowtrainer(num_cluster, terminate_criterion, 3, KMEANS_PP_CENTERS);
	bowtrainer.add(training_descriptors_S);

	cout << "Total descriptors: " << training_descriptors_S.rows << endl;

	cout << "-----------------\nTRAIN SET\n-----------------\n" << endl;


	vocabulary = bowtrainer.cluster();

	Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();
	BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	writeVocabulary("vocabulary_"+to_string(SMPLE_CLASS + 1)+"_"+to_string(num_cluster)+" dense_SIFT", vocabulary);
	*/



	Mat vocabulary;
	readVocabulary("vocabulary_102_200 dense_SIFT", vocabulary);

	Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();
	BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	Mat train_samples;
	Mat labels;

	cout << "\n------- SVM TRAIN ---------\n" << endl;


	/*
	Mat SVM_train_data(0, 0, CV_32FC1);
	Mat SVM_train_label(0, 1, CV_32FC1);
	Mat levelBowdescriptors[SPATIAL_LEVEL + 1];
	Mat levelBowdescriptor;
	Mat croppedBowdecriptor;
	Mat croppedImage;

	Size s;
	Size s1;
	int rows, cols, X, Y, width, height;

	struct _finddata_t fd_train;
	intptr_t handle_train;
	if ((handle_train = _findfirst(folderpath.c_str(), &fd_train)) == -1L) cout << "No file in directory!" << endl;

	int i = 0;
	int sample_cnt = 0;

	do {
		if (i >= 2)
		{
			vector<String> filenames;

			glob(folder_folderpath + fd_train.name, filenames);


			sample_cnt++;
			cout << fd_train.name << " : " << sample_cnt << endl;

			for (size_t j = 0; j < filenames.size(); j++)
			{
				if (filenames.size() <= 30) continue;
				if (j >= 30) continue;

				cout << ".....";

				vector< KeyPoint > Keypoint_img_train;

				Mat img_read = imread(filenames[j], IMREAD_GRAYSCALE);

				s = img_read.size();
				rows = s.height;
				cols = s.width;

				X = 0;
				Y = 0;
				width = cols / 2;
				height = rows / 2;

				// Level 0 to L
				for (int k = 0; k <= SPATIAL_LEVEL; k++)
				{
					width = cols / pow(2, k);
					height = rows / pow(2, k);

					Mat levelBowdescriptor;

					for (int m = 0; m < pow(2, 2 * k); m++)
					{
						// Set the left corner of subimage
						X = (m % (int)pow(2, k)) * width;
						Y = (m / (int)pow(2, k)) * height;


						// Get the subimage

						croppedImage = img_read(Rect(X, Y, width, height));

						Keypoint_img_train.clear();

						for (int m = STEP_SIZE; m < img_read.rows - STEP_SIZE; m += STEP_SIZE)
						{
							for (int n = STEP_SIZE; n < img_read.cols - STEP_SIZE; n += STEP_SIZE)
							{
								Keypoint_img_train.push_back(KeyPoint(float(n), float(m), float(STEP_SIZE)));
							}
						}

						if (Keypoint_img_train.empty()) cout << "No keypoints found." << endl;

						bowide.compute(img_read, Keypoint_img_train, croppedBowdecriptor);
						if (croppedBowdecriptor.empty()) cout << "No descriptors found." << endl;

						s1 = croppedBowdecriptor.size();

						if (s1.width == 0)
						{
							croppedBowdecriptor = Mat(1, 200, CV_32F, Scalar(0.));
							s1 = croppedBowdecriptor.size();
						}

						if (m == 0)
						{
							levelBowdescriptor = croppedBowdecriptor;
						}
						else
						{
							hconcat(levelBowdescriptor, croppedBowdecriptor, levelBowdescriptor);
						}
					}

					//Feature vectors of each levels
					levelBowdescriptors[k] = levelBowdescriptor;

				}

				if (SPM_MODE == 1)
				{
					SVM_train_data.push_back(levelBowdescriptors[SELECT_LEVE]);
					SVM_train_label.push_back(sample_cnt);
				}
				else if (SPM_MODE == 2)
				{

					Mat pyramid_levelBowdescriptor;

					for (int p = 0; p < SPATIAL_LEVEL + 1; p++)
					{
						float weight = ((float)pow(2, 2 - p));
						levelBowdescriptors[p] = levelBowdescriptors[p] / weight;

						if (p == 0)
						{
							pyramid_levelBowdescriptor = levelBowdescriptors[0];
						}
						else
						{
							hconcat(pyramid_levelBowdescriptor, levelBowdescriptors[p], pyramid_levelBowdescriptor);
						}
					}


					SVM_train_data.push_back(pyramid_levelBowdescriptor);
					SVM_train_label.push_back(sample_cnt);
				}

				cout << "\b\b\b\b\b";


			}
		}

		i++;

		if (sample_cnt > SMPLE_CLASS) break;

	} while (_findnext(handle_train, &fd_train) == 0);

	_findclose(handle_train);

	FileStorage fs1("SVM_train_data.yml", FileStorage::WRITE);
	fs1 << "SVM_train_data" << SVM_train_data;
	fs1.release();
	FileStorage fs2("SVM_train_label.yml", FileStorage::WRITE);
	fs2 << "SVM_train_label" << SVM_train_label;
	fs2.release();
	*/
	// Load the vocabulary from file.
	Mat SVM_train_data(0, 0, CV_32FC1);;
	FileStorage fs1("SVM_train_data.yml", FileStorage::READ);
	fs1["SVM_train_data"] >> SVM_train_data;
	fs1.release();
	Mat SVM_train_label(0, 1, CV_32FC1);;
	FileStorage fs2("SVM_train_label.yml", FileStorage::READ);
	fs2["SVM_train_label"] >> SVM_train_label;
	fs2.release();


	cout << "\n------- SVM TRAIN AUTO ---------\n" << endl;

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->setGamma(1.0);
	svm->setC(5.0);

	// Train the SVM with given parameters
	Ptr<TrainData> td = TrainData::create(SVM_train_data, ROW_SAMPLE, SVM_train_label);
	svm->trainAuto(td);

	cout << "\n------- TRAIN ---------\n" << endl;

	Mat result_train;
	float count_train = 0, accuracy_train = 0;

	svm->predict(SVM_train_data, result_train);

	for (int i = 0; i < SVM_train_label.rows; i++)
	{
		if (result_train.at<float>(i, 0) == SVM_train_label.at<int>(i, 0))
		{
			count_train = count_train + 1;
		}
	}

	accuracy_train = (count_train / SVM_train_label.rows) * 100;
	cout << "accuracy : " << accuracy_train << endl;


	cout << "\n------- TEST ---------\n" << endl;
	/*
	Mat test_data(0, 0, CV_32FC1);;
	Mat test_data_label(0, 1, CV_32FC1);;

	struct _finddata_t fd_test;
	intptr_t handle_test;
	if ((handle_test = _findfirst(folderpath.c_str(), &fd_test)) == -1L) cout << "No file in directory!" << endl;

	i = 0;
	sample_cnt = 0;

	do {
		if (i >= 2)
		{
			vector<String> filenames;

			glob(folder_folderpath + fd_test.name, filenames);

			sample_cnt++;
			cout << fd_test.name << " : " << sample_cnt << endl;


			for (size_t j = 0; j < filenames.size(); j++)
			{
				if (filenames.size() <= 30)
				{
					cout << ".....";

					vector< KeyPoint > Keypoint_img_test;

					Mat img_read = imread(filenames[j], IMREAD_GRAYSCALE);

					s = img_read.size();
					rows = s.height;
					cols = s.width;

					X = 0;
					Y = 0;
					width = cols / 2;
					height = rows / 2;

					// Level 0 to L
					for (int k = 0; k <= SPATIAL_LEVEL; k++)
					{
						width = cols / pow(2, k);
						height = rows / pow(2, k);

						Mat levelBowdescriptor;

						for (int m = 0; m < pow(2, 2 * k); m++)
						{
							// Set the left corner of subimage
							X = (m % (int)pow(2, k)) * width;
							Y = (m / (int)pow(2, k)) * height;


							// Get the subimage

							croppedImage = img_read(Rect(X, Y, width, height));

							Keypoint_img_test.clear();

							for (int m = STEP_SIZE; m < img_read.rows - STEP_SIZE; m += STEP_SIZE)
							{
								for (int n = STEP_SIZE; n < img_read.cols - STEP_SIZE; n += STEP_SIZE)
								{
									Keypoint_img_test.push_back(KeyPoint(float(n), float(m), float(STEP_SIZE)));
								}
							}

							if (Keypoint_img_test.empty()) cout << "No keypoints found." << endl;

							bowide.compute(img_read, Keypoint_img_test, croppedBowdecriptor);
							if (croppedBowdecriptor.empty()) cout << "No descriptors found." << endl;

							s1 = croppedBowdecriptor.size();

							if (s1.width == 0)
							{
								croppedBowdecriptor = Mat(1, 200, CV_32F, Scalar(0.));
								s1 = croppedBowdecriptor.size();
							}

							if (m == 0)
							{
								levelBowdescriptor = croppedBowdecriptor;
							}
							else
							{
								hconcat(levelBowdescriptor, croppedBowdecriptor, levelBowdescriptor);
							}
						}

						//Feature vectors of each levels
						levelBowdescriptors[k] = levelBowdescriptor;

					}

					if (SPM_MODE == 1)
					{
						test_data.push_back(levelBowdescriptors[SELECT_LEVE]);
						test_data_label.push_back(sample_cnt);
					}
					else if (SPM_MODE == 2)
					{
						Mat pyramid_levelBowdescriptor;
						for (int p = 0; p < SELECT_LEVE + 1; p++)
						{
							float weight = ((float)pow(2, 2 - p));
							levelBowdescriptors[p] = levelBowdescriptors[p] / weight;

							if (p == 0)
							{
								pyramid_levelBowdescriptor = levelBowdescriptors[0];
							}
							else
							{
								hconcat(pyramid_levelBowdescriptor, levelBowdescriptors[p], pyramid_levelBowdescriptor);
							}
						}
						test_data.push_back(pyramid_levelBowdescriptor);
						test_data_label.push_back(sample_cnt);
					}


					cout << "\b\b\b\b\b";
				}
				else if (j >= 30)
				{
					if (j >= 80) continue;

					cout << ".....";

					vector< KeyPoint > Keypoint_img_test;

					Mat img_read = imread(filenames[j],IMREAD_GRAYSCALE);

					s = img_read.size();
					rows = s.height;
					cols = s.width;

					X = 0;
					Y = 0;
					width = cols / 2;
					height = rows / 2;

					// Level 0 to L
					for (int k = 0; k <= SPATIAL_LEVEL; k++)
					{
						width = cols / pow(2, k);
						height = rows / pow(2, k);

						Mat levelBowdescriptor;

						for (int m = 0; m < pow(2, 2 * k); m++)
						{
							// Set the left corner of subimage
							X = (m % (int)pow(2, k)) * width;
							Y = (m / (int)pow(2, k)) * height;


							// Get the subimage

							croppedImage = img_read(Rect(X, Y, width, height));

							Keypoint_img_test.clear();

							for (int m = STEP_SIZE; m < img_read.rows - STEP_SIZE; m += STEP_SIZE)
							{
								for (int n = STEP_SIZE; n < img_read.cols - STEP_SIZE; n += STEP_SIZE)
								{
									Keypoint_img_test.push_back(KeyPoint(float(n), float(m), float(STEP_SIZE)));
								}
							}

							if (Keypoint_img_test.empty()) cout << "No keypoints found." << endl;

							bowide.compute(img_read, Keypoint_img_test, croppedBowdecriptor);
							if (croppedBowdecriptor.empty()) cout << "No descriptors found." << endl;

							s1 = croppedBowdecriptor.size();

							if (s1.width == 0)
							{
								croppedBowdecriptor = Mat(1, 200, CV_32F, Scalar(0.));
								s1 = croppedBowdecriptor.size();
							}

							if (m == 0)
							{
								levelBowdescriptor = croppedBowdecriptor;
							}
							else
							{
								hconcat(levelBowdescriptor, croppedBowdecriptor, levelBowdescriptor);
							}
						}

						//Feature vectors of each levels
						levelBowdescriptors[k] = levelBowdescriptor;

					}

					Mat kernels;
					//Weight the features of level according to the equation given in paper.
					for (int i = 0; i <= SPATIAL_LEVEL; i++)
					{
						s = kernels.size();

						levelBowdescriptor = levelBowdescriptors[i];
						float weight = ((float)pow(2, 2 - i));
						levelBowdescriptor = levelBowdescriptor / weight;

						if (s.width == 0 && s.height == 0) kernels = levelBowdescriptor;
						else hconcat(kernels, levelBowdescriptor, kernels);

					}


					if (SPM_MODE == 1)
					{
						test_data.push_back(levelBowdescriptors[SELECT_LEVE]);
						test_data_label.push_back(sample_cnt);
					}
					else if (SPM_MODE == 2)
					{
						test_data.push_back(kernels);
						test_data_label.push_back(sample_cnt);
					}

					cout << "\b\b\b\b\b";
				}

			}
		}

		i++;

		if (sample_cnt > SMPLE_CLASS) break;

	} while (_findnext(handle_test, &fd_test) == 0);

	_findclose(handle_test);

	FileStorage fs3("test_data.yml", FileStorage::WRITE);
	fs3 << "test_data" << test_data;
	fs3.release();
	FileStorage fs4("test_data_label.yml", FileStorage::WRITE);
	fs4 << "test_data_label" << test_data_label;
	fs4.release();
	*/

	// Load the vocabulary from file.
	Mat test_data(0, 0, CV_32FC1);;
	FileStorage fs3("test_data.yml", FileStorage::READ);
	fs3["test_data"] >> test_data;
	fs3.release();
	Mat test_data_label(0, 1, CV_32FC1);;
	FileStorage fs4("test_data_label.yml", FileStorage::READ);
	fs4["test_data_label"] >> test_data_label;
	fs4.release();

	Mat result_test;
	float count_test = 0, accuracy_test = 0;

	svm->predict(test_data, result_test);

	for (int i = 0; i < test_data_label.rows; i++)
	{
		if (result_test.at<float>(i, 0) == test_data_label.at<int>(i, 0))
		{
			count_test = count_test + 1;
		}
	}


	accuracy_test = (count_test / test_data_label.rows) * 100;

	cout << "\naccuracy : " << accuracy_test << endl;


	cout << "-------SVM_TRAIN SAVE ---------\n";
	cout << "\n\n";

	svm->save("./svm/svm_Class" + to_string(SMPLE_CLASS + 1) + "_spm_mode_" + to_string(SPM_MODE) + "_level_" + to_string(SELECT_LEVE) + "_test_acc" + to_string(accuracy_test) + "_train_acc" + to_string(accuracy_train) + "_FLANN");

	return 0;


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
