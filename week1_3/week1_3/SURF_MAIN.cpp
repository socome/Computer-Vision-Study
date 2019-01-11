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

	    // ���� �̹��� �ҷ�����
		Mat img1;
		img1 = imread("object.jpg", IMREAD_COLOR);
		if (img1.empty())
		{
			cout << "Couldn't load" << endl;
			return -1;
		}

		//���� Ư¡�� �м��� ���� GRAY ��ȯ 1
		Mat matGrayImage;
		cvtColor(img1, matGrayImage, CV_RGB2GRAY);

		//SURF �غ�
		Ptr<SurfFeatureDetector> Detector = SURF::create(nMinHessian);
		Ptr<SurfDescriptorExtractor> Extractor = SURF::create();
		vector<KeyPoint> vtKeypointsImage, vtKeypointVideo;

		//ã�����ϴ� ��ü(����)�� SURF�� ���� Ư¡�� ����&���
		Mat matDescriptorsImage;
		Detector->detect(img1, vtKeypointsImage);
		Extractor->compute(img1, vtKeypointsImage, matDescriptorsImage);

		// ��Ʈ�� ī�޶� �غ� �ڵ�
		VideoCapture cap1(0);

		if (!cap1.isOpened())
		{
			cout << "Cam fails to launch " << endl;
			return -1;
		}
		Mat video_frame;

		for (;;)
		{
			//���� �̹����� video_frame�� ����
			cap1 >> video_frame;

			//���� Ư¡�� �м��� ���� GRAY ��ȯ 2 
			Mat matGrayvideo;
			cvtColor(video_frame, matGrayvideo, CV_RGB2GRAY);

			if (!matGrayImage.data || !matGrayvideo.data)
			{
				std::cout << "Gray fail" << std::endl;
				return -1;
			}

			//�ǽð� ���� �������� SURF�� ���� Ư¡�� ����&���
			Detector->detect(video_frame, vtKeypointVideo);
			Mat matDescriptorsVideo;
			Extractor->compute(video_frame, vtKeypointVideo, matDescriptorsVideo);


			//FLANN ���� �ٻ� ���� �̿� Ž�� �� �̿��� ��Ī�Ѵ� 
			double dMaxDist = 0;
			double dMinDist = 100;
			double dDistance;

			FlannBasedMatcher Matcher;
			vector<DMatch> matches;
			Matcher.match(matDescriptorsImage, matDescriptorsVideo, matches);

			// �� ���� keypoint ���̿��� min-max�� ����Ѵ�. -> �ʹ� ���� ��Ī���� ���� ��Ȯ�� ��Ī�� �ϱ����ؼ�!! 
			// �ٸ� ��ΰŴ� �Ÿ��� �����ϱ⵵ �ϴ��� .. ã�ƺ� �ʿ� ����


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
