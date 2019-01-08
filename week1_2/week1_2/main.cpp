#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <stdio.h>
#include <iostream>


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

int main()
{
	Mat matLeftImage;
	Mat matRightImage;
	Mat matGrayLImage;
	Mat matGrayRImage;

	matLeftImage = imread("image_l.jpg", IMREAD_COLOR);
	matRightImage = imread("image_r.jpg", IMREAD_COLOR);

	if (matLeftImage.empty() || matRightImage.empty())
	{
		std::cout << "image load fail" << std::endl;
		return -1;
	}
	// 이미지 down-sampling

	//Size size(matLeftImage.cols / 2, matLeftImage.rows / 2);
	//resize(matLeftImage, matLeftImage, size);
	//resize(matRightImage, matRightImage, size);

	// Gray 이미지 변환 -> 특징점을 빠르게 찾기위함

	cvtColor(matLeftImage, matGrayLImage, CV_RGB2GRAY);
	cvtColor(matRightImage, matGrayRImage, CV_RGB2GRAY);

	if (!matGrayLImage.data || !matGrayRImage.data)
	{
		std::cout << "Gray fail" << std::endl;
		return -1;
	}

	//step1 : SURF Detecctor를 이용해 키포인트 찾기
	double nMinHessian = 400.;  //thresold

	Ptr<SurfFeatureDetector> Detector = SURF::create(nMinHessian);

	vector< KeyPoint > vtKeypointsObject, vtKeypointScene;

	// SURF를 이용해 추출퇸 특징값을 KeyPoint에 저장
	Detector->detect(matGrayLImage, vtKeypointsObject);
	Detector->detect(matGrayRImage, vtKeypointScene);

	//step2 : Calculate Descriptors (feature vector) 점에대한 정보 수집
	Ptr<SurfDescriptorExtractor> Extractor = SURF::create();

	Mat matDescriptorsObject, matDescriptorsScene;

	Extractor->compute(matGrayLImage, vtKeypointsObject, matDescriptorsObject);
	Extractor->compute(matGrayRImage, vtKeypointScene, matDescriptorsScene);

	//step3 : Decriptor를 이용해 FLANN을 매칭한다 (all 매칭)
	
	FlannBasedMatcher Matcher;
	vector<DMatch> matches;
	Matcher.match(matDescriptorsObject, matDescriptorsScene, matches);

	Mat matGoodMatches1;
	drawMatches(matGrayLImage, vtKeypointsObject, matGrayRImage, vtKeypointScene, matches, matGoodMatches1, Scalar::all(-1), Scalar(-1), vector<char>(),DrawMatchesFlags::DEFAULT);
	imshow("all-matches", matGoodMatches1);

	double dMaxDist = 0;
	double dMinDist = 100;
	double dDistance;

	// 두 개의 keypoint 사이에서 min-max를 계산한다.
	for (int i = 0; i < matDescriptorsObject.rows; i++)
	{
		dDistance = matches[i].distance;

		if (dDistance < dMinDist) dMinDist = dDistance;
		if (dDistance > dMaxDist) dMaxDist = dDistance;

	}

	printf("-- Max iddst : %f \n", dMaxDist);
	printf("-- Min iddst : %f \n", dMinDist);

	// good maches만 사용한다. (all 매치 아님)

	vector<DMatch> good_matches;

	for (int i = 0; i < matDescriptorsObject.rows; i++)
	{
		if (matches[i].distance < 5 * dMinDist)
		{
			good_matches.push_back(matches[i]);
		}
	}
	
	Mat matGoodMatcges;
	drawMatches(matGrayLImage, vtKeypointsObject, matGrayRImage, vtKeypointScene, good_matches, matGoodMatcges, Scalar::all(-1), Scalar(-1), vector<char>(), DrawMatchesFlags::DEFAULT);
	imshow("good-matches", matGoodMatcges);
	//isolate the matched keypotints in each image

	vector<Point2f> obj;
	vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		obj.push_back(vtKeypointsObject[good_matches[i].queryIdx].pt);
		scene.push_back(vtKeypointScene[good_matches[i].trainIdx].pt);
	}

	Mat HomoMatrix = findHomography(scene, obj, CV_RANSAC);
	
	cout << HomoMatrix << endl;

	// homoMatrix를 사용하여 이미지를 warp

	Mat matResult;

	warpPerspective(matRightImage, matResult, HomoMatrix, Size(matRightImage.cols * 2, matRightImage.rows), INTER_CUBIC);
	
	Mat matPanorama;
	matPanorama = matResult.clone();

	imshow("warp", matResult);
	

	Mat matROI(matPanorama,Rect(0, 0, matLeftImage.cols, matLeftImage.rows));
	matLeftImage.copyTo(matROI);

	imshow("panorama", matPanorama);

	waitKey(0);


	//검정색 배경화면 제거

	//vector<Point> nonBlackList;
	//nonBlackList.reserve(matResult.rows  * matResult.clos);

	return 0;
}