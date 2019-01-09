
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
	// �̹��� down-sampling

	//Size size(matLeftImage.cols / 2, matLeftImage.rows / 2);
	//resize(matLeftImage, matLeftImage, size);
	//resize(matRightImage, matRightImage, size);

	// Gray �̹��� ��ȯ -> Ư¡���� ������ ã������

	cvtColor(matLeftImage, matGrayLImage, CV_RGB2GRAY);
	cvtColor(matRightImage, matGrayRImage, CV_RGB2GRAY);

	if (!matGrayLImage.data || !matGrayRImage.data)
	{
		std::cout << "Gray fail" << std::endl;
		return -1;
	}

	//step1 : SURF Detecctor�� �̿��� Ű����Ʈ ã��
	double nMinHessian = 400.;  //thresold

	Ptr<SurfFeatureDetector> Detector = SURF::create(nMinHessian);

	vector< KeyPoint > vtKeypointsObject, vtKeypointScene;

	// SURF�� �̿��� ������ Ư¡���� KeyPoint�� ����
	Detector->detect(matGrayLImage, vtKeypointsObject);
	Detector->detect(matGrayRImage, vtKeypointScene);

	//step2 : Calculate Descriptors (feature vector) �������� ���� ����
	Ptr<SurfDescriptorExtractor> Extractor = SURF::create();

	Mat matDescriptorsObject, matDescriptorsScene;

	Extractor->compute(matGrayLImage, vtKeypointsObject, matDescriptorsObject);
	Extractor->compute(matGrayRImage, vtKeypointScene, matDescriptorsScene);

	//step3 : Decriptor�� �̿��� FLANN�� ��Ī�Ѵ� (all ��Ī)
	
	FlannBasedMatcher Matcher;
	vector<DMatch> matches;
	Matcher.match(matDescriptorsObject, matDescriptorsScene, matches);

	Mat matGoodMatches1;
	drawMatches(matGrayLImage, vtKeypointsObject, matGrayRImage, vtKeypointScene, matches, matGoodMatches1, Scalar::all(-1), Scalar(-1), vector<char>(),DrawMatchesFlags::DEFAULT);
	imshow("all-matches", matGoodMatches1);

	double dMaxDist = 0;
	double dMinDist = 100;
	double dDistance;

	// �� ���� keypoint ���̿��� min-max�� ����Ѵ�.
	for (int i = 0; i < matDescriptorsObject.rows; i++)
	{
		dDistance = matches[i].distance;

		if (dDistance < dMinDist) dMinDist = dDistance;
		if (dDistance > dMaxDist) dMaxDist = dDistance;

	}

	printf("-- Max iddst : %f \n", dMaxDist);
	printf("-- Min iddst : %f \n", dMinDist);

	// good maches�� ����Ѵ�. (all ��ġ �ƴ�)

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

	// homoMatrix�� ����Ͽ� �̹����� warp

	Mat matResult;

	warpPerspective(matRightImage, matResult, HomoMatrix, Size(matRightImage.cols * 2, matRightImage.rows), INTER_CUBIC);
	
	Mat matPanorama;
	matPanorama = matResult.clone();

	imshow("warp", matResult);
	

	Mat matROI(matPanorama,Rect(0, 0, matLeftImage.cols, matLeftImage.rows));
	matLeftImage.copyTo(matROI);

	imshow("panorama", matPanorama);

	waitKey(0);


	//������ ���ȭ�� ����

	//vector<Point> nonBlackList;
	//nonBlackList.reserve(matResult.rows  * matResult.clos);

	return 0;
}
