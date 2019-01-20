/*
#include <stdio.h>
#include <iostream>

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

int panorama_t()
{
	// Mat,vector 등 초기 변수(?) 설정

	Mat matLeftImage;
	Mat matRightImage;
	Mat matGrayLImage;
	Mat matGrayRImage;
	Mat matDescriptorsObject, matDescriptorsScene;
	Mat matallMatches;
	Mat matGoodMatches;
	Mat matResult;
	Mat matPanorama;


	vector< KeyPoint > vtKeypointsObject, vtKeypointScene;
	vector< DMatch > matches;
	vector<DMatch> good_matches;
	vector<Point2f> obj;
	vector<Point2f> scene;
	vector<Point> nonBlackList;


	double nMinHessian = 400.;
	int imagewidth = 0;
	int imageheigh = 0;
	double dMaxDist = 0;
	double dMinDist = 100;
	double dDistance;

	//붙여올 사진 2장 불러오기&체크
	matLeftImage = imread("image1.jpg", IMREAD_COLOR);
	matRightImage = imread("image1.jpg", IMREAD_COLOR);

	imagewidth = matLeftImage.cols * 2;
	imageheigh = matLeftImage.rows * 2;

	if (matLeftImage.empty() || matRightImage.empty())
	{
		std::cout << "image load fail" << std::endl;
		return -1;
	}

	//특이점 추출을 더욱 잘하기 위해 이미지 변환(흑백) & 체크
	cvtColor(matLeftImage, matGrayLImage, CV_RGB2GRAY);
	cvtColor(matRightImage, matGrayRImage, CV_RGB2GRAY);
	if (!matGrayLImage.data || !matGrayRImage.data)
	{
		std::cout << "Gray fail" << std::endl;
		return -1;
	}

	//SIFT 기법을 이용한 특징점 검출
	Ptr<SiftFeatureDetector> Detector = SIFT::create(nMinHessian);
	Detector->detect(matGrayLImage, vtKeypointsObject);
	Detector->detect(matGrayRImage, vtKeypointScene);
	Ptr<SiftDescriptorExtractor> Extractor = SIFT::create();
	Extractor->compute(matGrayLImage, vtKeypointsObject, matDescriptorsObject);
	Extractor->compute(matGrayRImage, vtKeypointScene, matDescriptorsScene);

	//SURF 기법을 이용한 특징점 검출
	//Ptr<SurfFeatureDetector> Detector = SURF::create(nMinHessian);
	//Detector->detect(matGrayLImage, vtKeypointsObject);
	//Detector->detect(matGrayRImage, vtKeypointScene);
	//Ptr<SurfFeatureDetector> Extractor = SURF::create();
	//Extractor->compute(matGrayLImage, vtKeypointsObject, matDescriptorsObject);
	//Extractor->compute(matGrayRImage, vtKeypointScene, matDescriptorsScene);


	//FlannBasedMatcher(빠른 근사 근접 이웃 탐색) 사용
	FlannBasedMatcher Matcher;
	Matcher.match(matDescriptorsObject, matDescriptorsScene, matches);

	//BFMatch 사용
	//BFMatcher matcher(4);
	//matcher.match(matDescriptorsObject, matDescriptorsScene, matches);


	//현재까지 과정을 imshow로 나타냄
	drawMatches(matGrayLImage, vtKeypointsObject, matGrayRImage, vtKeypointScene, matches, matallMatches, Scalar::all(-1), Scalar(-1), vector<char>(), DrawMatchesFlags::DEFAULT);
	imshow("all-matches", matallMatches);


	//좋은 매칭 걸러내기 step1. 매칭점수(거리) 계산
	for (int i = 0; i < matDescriptorsObject.rows; i++)
	{
		dDistance = matches[i].distance;

		if (dDistance < dMinDist) dMinDist = dDistance;
		if (dDistance > dMaxDist) dMaxDist = dDistance;

	}
	printf("-- Max dist : %f \n", dMaxDist);
	printf("-- Min disst : %f \n", dMinDist);



	for (int i = 0; i < matDescriptorsObject.rows; i++)
	{
		if (matches[i].distance < 5 * dMinDist)
		{
			good_matches.push_back(matches[i]);
		}
	}


	drawMatches(matGrayLImage, vtKeypointsObject, matGrayRImage, vtKeypointScene, good_matches, matGoodMatches, Scalar::all(-1), Scalar(-1), vector<char>(), DrawMatchesFlags::DEFAULT);
	imshow("good-matches", matGoodMatches);


	for (int i = 0; i < good_matches.size(); i++)
	{
		obj.push_back(vtKeypointsObject[good_matches[i].queryIdx].pt);
		scene.push_back(vtKeypointScene[good_matches[i].trainIdx].pt);
	}

	Mat HomoMatrix = findHomography(scene, obj, CV_RANSAC, 5);

	cout << HomoMatrix << endl;

	// homoMatrix를 사용하여 이미지를 warp
	warpPerspective(matRightImage, matResult, HomoMatrix, Size(imagewidth , imageheigh), INTER_CUBIC);

	matPanorama = matResult.clone();
	imshow("warp", matResult);


	Mat matROI(matPanorama, Rect(0, 0, matLeftImage.cols, matLeftImage.rows));
	matLeftImage.copyTo(matROI);

	imshow("panorama", matPanorama);


	nonBlackList.reserve(matResult.rows *matResult.cols);

	//   add all non-black points to the vector
	//   there are more efficient ways to iterate through the image
	for (int j = 0; j < matPanorama.rows; ++j)
	{
		for (int i = 0; i < matPanorama.cols; ++i)
		{
			//   if not black: add to the list
			if (matPanorama.at<Vec3b>(j, i) != Vec3b(0, 0, 0))
			{
				nonBlackList.push_back(Point(i, j));
			}
		}
	}

	//   create bounding rect around those points
	Rect rtBound = boundingRect(nonBlackList);

	matPanorama = matPanorama(rtBound);
	//   결과를 display하고 파일로 출력한다.
	imshow("Result", matPanorama);


	waitKey(0);

}
*/

/* 백업 메인


*/