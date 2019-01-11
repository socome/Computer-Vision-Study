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

double nMinHessian = 1000.;  //thresold


int main2()
	{

	    // 비교할 이미지 불러오기
		Mat img1;
		img1 = imread("object.jpg", IMREAD_COLOR);
		if (img1.empty())
		{
			cout << "Couldn't load" << endl;
			return -1;
		}

		//빠른 특징점 분석을 위한 GRAY 변환 1
		Mat matGrayImage;
		cvtColor(img1, matGrayImage, CV_RGB2GRAY);

		//SURF 준비
		Ptr<SurfFeatureDetector> Detector = SURF::create(nMinHessian);
		Ptr<SurfDescriptorExtractor> Extractor = SURF::create();
		vector<KeyPoint> vtKeypointsImage, vtKeypointVideo;

		//찾고자하는 물체(사진)을 SURF를 통한 특징점 추출&계산
		Mat matDescriptorsImage;
		Detector->detect(img1, vtKeypointsImage);
		Extractor->compute(img1, vtKeypointsImage, matDescriptorsImage);

		// 노트북 카메라 준비 코드
		VideoCapture cap1(0);

		if (!cap1.isOpened())
		{
			cout << "Cam fails to launch " << endl;
			return -1;
		}
		Mat video_frame;

		for (;;)
		{
			//비디오 이미지를 video_frame에 저장
			cap1 >> video_frame;

			//빠른 특징점 분석을 위한 GRAY 변환 2 
			Mat matGrayvideo;
			cvtColor(video_frame, matGrayvideo, CV_RGB2GRAY);

			if (!matGrayImage.data || !matGrayvideo.data)
			{
				std::cout << "Gray fail" << std::endl;
				return -1;
			}

			//실시간 비디오 프레임을 SURF를 통한 특징점 추출&계산
			Detector->detect(video_frame, vtKeypointVideo);
			Mat matDescriptorsVideo;
			Extractor->compute(video_frame, vtKeypointVideo, matDescriptorsVideo);


			//FLANN 빠른 근사 근접 이웃 탐색 을 이용해 매칭한다 
			double dMaxDist = 0;
			double dMinDist = 100;
			double dDistance;

			FlannBasedMatcher Matcher;
			vector<DMatch> matches;
			Matcher.match(matDescriptorsImage, matDescriptorsVideo, matches);

			// 두 개의 keypoint 사이에서 min-max를 계산한다. -> 너무 많은 매칭에서 좀더 정확한 매칭을 하기위해서!! 
			// 다른 블로거는 거리를 정의하기도 하던데 .. 찾아볼 필요 있음


				for (int i = 0; i < matDescriptorsImage.rows; i++)
				{
					dDistance = matches[i].distance;

					if (dDistance < dMinDist) dMinDist = dDistance;
					if (dDistance > dMaxDist) dMaxDist = dDistance;

				}

				printf("-- Max iddst : %f \n", dMaxDist);
				printf("-- Min iddst : %f \n", dMinDist);

			vector<DMatch> good_matches;

			for (int i = 0; i < matDescriptorsImage.rows; i++)
			{
				if (matches[i].distance <= max(2* dMinDist,0.02))
				{
					good_matches.push_back(matches[i]);
				}
			}

			Mat matGoodMatcges;
			drawMatches(matGrayImage, vtKeypointsImage, matGrayvideo, vtKeypointVideo, good_matches, matGoodMatcges, Scalar::all(-1), Scalar(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

			namedWindow("surf matches", 0);
			imshow("surf matches", matGoodMatcges);

			if (waitKey(30) >= 0) break;

		}



	



			
		return EXIT_SUCCESS;

		
	}
