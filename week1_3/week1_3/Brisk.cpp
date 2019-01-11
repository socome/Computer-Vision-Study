#include <iostream>
#include <stdio.h>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


int main()
{
	int nMinHessian = 300.;  //thresold
	double dMaxDist = 0;
	double dMinDist = 100;
	double dDistance;

	Mat image;
	Mat matGrayImage;
	Mat matDescriptorsImage;
	Mat video_frame;
	Mat video_frame_bef;
	Mat matGrayvideo;
	Mat matDescriptorsVideo;
	Mat matGoodMatcges;

	vector<KeyPoint> vtKeypointsImage, vtKeypointVideo;
	vector<DMatch> matches;
	vector<DMatch> good_matches;

	vector<Point2f> image_pt;
	vector<Point2f> video_pt;
	vector<Point2f> image_coner(4);
	vector<Point2f> video_coner(4);

	//video line�� �׸��� ���� color scalar
	Scalar COLOR(255, 0, 0);

	VideoCapture test_video;

	// ���� �̹��� �ҷ�����
	image = imread("object.jpg", IMREAD_COLOR);
	if (image.empty())
	{
		cout << "Couldn't load" << endl;
		return -1;
	}

	//���� Ư¡�� �м��� ���� GRAY ��ȯ 1
	cvtColor(image, matGrayImage, CV_RGB2GRAY);

	//SIFT �غ�
	//Ptr<SiftFeatureDetector> Detector = SIFT::create(nMinHessian);
	//Ptr<SiftDescriptorExtractor> Extractor = SIFT::create();


	//BRISK�� �̹����� ����� Ư¡�� ����& ���
	Ptr<BRISK> Brisk = BRISK::create(30, 3, 1.0f);
	Brisk->detectAndCompute(matGrayImage, noArray(), vtKeypointsImage, matDescriptorsImage);


	// ��Ʈ�� ī�޶� �غ� �ڵ�
	//VideoCapture cap1(0);
	//�ǽð� ī�޶� �������� ���ϸ� test_video -> cap1���� ����

	//������ �ҷ�����
	test_video.open("test_video.mp4");

	if (!test_video.isOpened())
	{
		cout << "Cam fails to launch " << endl;
		return -1;
	}


	//ã�� image �ڳ��� �м�
	image_coner[0] = cvPoint(0, 0);
	image_coner[1] = cvPoint(image.cols, 0);
	image_coner[2] = cvPoint(image.cols, image.rows);
	image_coner[3] = cvPoint(0, image.rows);


	for (;;)
	{
		//���� �̹����� video_frame�� ����
		test_video >> video_frame_bef;
		flip(video_frame_bef, video_frame, -1);

		//���� Ư¡�� �м��� ���� GRAY ��ȯ 2 
		cvtColor(video_frame, matGrayvideo, CV_RGB2GRAY);

		if (!matGrayImage.data || !matGrayvideo.data)
		{
			std::cout << "Gray fail" << std::endl;
			return -1;
		}

		//BRISK�� ���� �������� ����� Ư¡�� ����& ���
		Brisk->detectAndCompute(matGrayvideo, noArray(), vtKeypointVideo, matDescriptorsVideo);

		//
		BFMatcher matcher(NORM_HAMMING);
		matcher.match(matDescriptorsImage, matDescriptorsVideo, matches);


		for (int i = 0; i < matDescriptorsImage.rows; i++)
		{
			dDistance = matches[i].distance;

			if (dDistance < dMinDist) dMinDist = dDistance;
			if (dDistance > dMaxDist) dMaxDist = dDistance;

		}

		printf("-- Max iddst : %f \n", dMaxDist);
		printf("-- Min iddst : %f \n", dMinDist);


		for (int i = 0; i < matDescriptorsImage.rows; i++)
		{
			if (matches[i].distance <= max(2 * dMinDist, 0.2))
			{
				good_matches.push_back(matches[i]);
			}
		}


		drawMatches(image, vtKeypointsImage, video_frame, vtKeypointVideo, good_matches, matGoodMatcges, Scalar::all(-1), Scalar(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		// RANSAC ���

		for (int i = 0; i < good_matches.size(); i++)
		{
			image_pt.push_back(vtKeypointsImage[good_matches[i].queryIdx].pt);
			video_pt.push_back(vtKeypointVideo[good_matches[i].trainIdx].pt);
		}

		Mat H = findHomography(image_pt, video_pt, CV_RANSAC);

		perspectiveTransform(image_coner, video_coner, H);


		Point2f p(image.cols, 0);
		// Video image�� �簢�� line ��Ÿ����
		line(matGoodMatcges, video_coner[0] + p, video_coner[1] + p, COLOR, 2);
		line(matGoodMatcges, video_coner[1] + p, video_coner[2] + p, COLOR, 2);
		line(matGoodMatcges, video_coner[2] + p, video_coner[3] + p, COLOR, 2);
		line(matGoodMatcges, video_coner[3] + p, video_coner[0] + p, COLOR, 2);

		namedWindow("brisk matches", WINDOW_NORMAL);
		imshow("brisk matches", matGoodMatcges);

		if (waitKey(30) >= 0) break;

	}


	return EXIT_SUCCESS;


}
