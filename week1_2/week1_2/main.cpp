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

int find_matches(Mat img1, Mat img2, int feature_N)
{
	if (img1.empty() || img2.empty())
	{
		std::cout << "image load fail" << std::endl;
	}

	vector< KeyPoint > vtKeypoint_img1, vtKeypoint_img2;
	Mat img1_g, img2_g;
	Mat img1_descriptors, img2_descriptors;

	cvtColor(img1, img1_g, CV_RGB2GRAY);
	cvtColor(img2, img2_g, CV_RGB2GRAY);

	if (!img1.data || !img2.data)
	{
		std::cout << "Gray fail" << std::endl;
	}

	//SIFT 기법을 이용한 특징점 검출
	Ptr<SiftFeatureDetector> Detector = SIFT::create(feature_N);
	Detector->detect(img1_g, vtKeypoint_img1);
	Detector->detect(img2_g, vtKeypoint_img2);
	Ptr<SiftDescriptorExtractor> Extractor = SIFT::create();
	Extractor->compute(img1_g, vtKeypoint_img1, img1_descriptors);
	Extractor->compute(img2_g, vtKeypoint_img2, img2_descriptors);

	double dMaxDist = 0;
	double dMinDist = 100;
	double dDistance;

	vector< DMatch > matches;
	vector<DMatch> good_matches;
	FlannBasedMatcher Matcher;
	Matcher.match(img1_descriptors, img2_descriptors, matches);


	//Mat matallMatches;
	//현재까지 과정을 imshow로 나타냄
	//drawMatches(img1_g, vtKeypoint_img1, img2_g, vtKeypoint_img2, matches, matallMatches, Scalar::all(-1), Scalar(-1), vector<char>(), DrawMatchesFlags::DEFAULT);
	//imshow("all-matches", matallMatches);


	for (int i = 0; i < img1_descriptors.rows; i++)
	{
		dDistance = matches[i].distance;

		if (dDistance < dMinDist) dMinDist = dDistance;
		if (dDistance > dMaxDist) dMaxDist = dDistance;

	}

	for (int i = 0; i < img1_descriptors.rows; i++)
	{
		if (matches[i].distance < 5 * dMinDist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	vector<Point2f> img1_pt;
	vector<Point2f> img2_pt;

	for (int i = 0; i < good_matches.size(); i++)
	{
		img1_pt.push_back(vtKeypoint_img1[good_matches[i].queryIdx].pt);
		img2_pt.push_back(vtKeypoint_img2[good_matches[i].trainIdx].pt);
	}

	Mat mask;
	Mat HomoMatrix = findHomography(img2_pt, img1_pt, CV_RANSAC, 5, mask);

	double outline_cnt = 0;
	double inline_cnt = 0;

	for (int i = 0; i < mask.rows; i++)
	{
		if (mask.at<bool>(i) == 0)
		{
			outline_cnt++;
		}
		else
		{
			inline_cnt++;
		}
	}

	double percentage = ((inline_cnt)/(inline_cnt+ outline_cnt))*100;
	int match_flag = 0;

	if (percentage >= 10 )
	{
		match_flag = 1;
	}
	std::cout << "percentage : " << percentage << std::endl;
	
	return match_flag;
}

Mat sift_panorama(Mat img1, Mat img2, int feature_N)
{
	if (img1.empty() || img2.empty())
	{
		std::cout << "image load fail" << std::endl;
	}

	vector< KeyPoint > vtKeypoint_img1, vtKeypoint_img2;
	Mat img1_g, img2_g;
	Mat img1_descriptors, img2_descriptors;

	cvtColor(img1, img1_g, CV_RGB2GRAY);
	cvtColor(img2, img2_g, CV_RGB2GRAY);

	if (!img1.data || !img2.data)
	{
		std::cout << "Gray fail" << std::endl;
	}

	//SIFT 기법을 이용한 특징점 검출
	Ptr<SiftFeatureDetector> Detector = SIFT::create(feature_N);
	Detector->detect(img1_g, vtKeypoint_img1);
	Detector->detect(img2_g, vtKeypoint_img2);
	Ptr<SiftDescriptorExtractor> Extractor = SIFT::create();
	Extractor->compute(img1_g, vtKeypoint_img1, img1_descriptors);
	Extractor->compute(img2_g, vtKeypoint_img2, img2_descriptors);


	int constant_K = 1;
	vector<DMatch> good_matches;

	do
	{
		good_matches.clear();
		constant_K++;

		vector< DMatch > matches;
		FlannBasedMatcher Matcher;
		Matcher.match(img1_descriptors, img2_descriptors, matches);

		/*
		//BFMatch 사용
		//BFMatcher matcher(4);
		//matcher.match(matDescriptorsObject, matDescriptorsScene, matches);
		*/


		double dMaxDist = 0;
		double dMinDist = 100;
		double dDistance;

		//좋은 매칭 걸러내기 
		for (int i = 0; i < img1_descriptors.rows; i++)
		{
			dDistance = matches[i].distance;

			if (dDistance < dMinDist) dMinDist = dDistance;
			if (dDistance > dMaxDist) dMaxDist = dDistance;

		}
		std::cout << "Max :" << dMaxDist << endl;
		std::cout << "Min :" << dMinDist << endl;

		for (int i = 0; i < img1_descriptors.rows; i++)
		{
			if (matches[i].distance < constant_K * dMinDist)
			{
				good_matches.push_back(matches[i]);
			}
		}

		if (constant_K > 7)
		{
			std::cout << "feature_N 값을 변경하세요" << endl;
			break;
		}
	} while (good_matches.size() < 20);

	
	std::cout << "Good Match : " << good_matches.size() << std::endl;
	std::cout << "constant K : " << constant_K << std::endl;

	vector<Point2f> img1_pt;
	vector<Point2f> img2_pt;

	for (int i = 0; i < good_matches.size(); i++)
	{
		img1_pt.push_back(vtKeypoint_img1[good_matches[i].queryIdx].pt);
		img2_pt.push_back(vtKeypoint_img2[good_matches[i].trainIdx].pt);
	}

	Mat HomoMatrix = findHomography(img2_pt, img1_pt, CV_RANSAC, 5);

	std::cout << HomoMatrix << endl;
	
	Mat matResult;
	Mat matPanorama;

	// 4개의 코너 구하기
	vector<Point2f> conerPt;

	conerPt.push_back(Point2f(0, 0));
	conerPt.push_back(Point2f(img1.size().width, 0));
	conerPt.push_back(Point2f(0, img1.size().height));
	conerPt.push_back(Point2f(img1.size().width, img1.size().height));

	Mat P_Trans_conerPt;
	perspectiveTransform(Mat(conerPt), P_Trans_conerPt, HomoMatrix);

	// 이미지의 모서리 계산
	double min_x, min_y, max_x, max_y;
	float min_x1, min_x2, min_y1, min_y2, max_x1, max_x2, max_y1, max_y2;

	min_x1 = min(P_Trans_conerPt.at<Point2f>(0).x, P_Trans_conerPt.at<Point2f>(1).x);
	min_x2 = min(P_Trans_conerPt.at<Point2f>(2).x, P_Trans_conerPt.at<Point2f>(3).x);
	min_y1 = min(P_Trans_conerPt.at<Point2f>(0).y, P_Trans_conerPt.at<Point2f>(1).y);
	min_y2 = min(P_Trans_conerPt.at<Point2f>(2).y, P_Trans_conerPt.at<Point2f>(3).y);
	max_x1 = max(P_Trans_conerPt.at<Point2f>(0).x, P_Trans_conerPt.at<Point2f>(1).x);
	max_x2 = max(P_Trans_conerPt.at<Point2f>(2).x, P_Trans_conerPt.at<Point2f>(3).x);
	max_y1 = max(P_Trans_conerPt.at<Point2f>(0).y, P_Trans_conerPt.at<Point2f>(1).y);
	max_y2 = max(P_Trans_conerPt.at<Point2f>(2).y, P_Trans_conerPt.at<Point2f>(3).y);
	min_x = min(min_x1, min_x2);
	min_y = min(min_y1, min_y2);
	max_x = max(max_x1, max_x2);
	max_y = max(max_y1, max_y2);

	// Transformation matrix
	Mat Htr = Mat::eye(3, 3, CV_64F);
	if (min_x < 0) {
		max_x = img2.size().width - min_x;
		Htr.at<double>(0, 2) = -min_x;
	}
	if (min_y < 0) {
		max_y = img2.size().height - min_y;
		Htr.at<double>(1, 2) = -min_y;
	}

	// 파노라마 만들기
	matPanorama = Mat(Size(max_x, max_y), CV_32F);
	warpPerspective(img1, matPanorama, Htr, matPanorama.size(), INTER_CUBIC, BORDER_CONSTANT, 0);
	warpPerspective(img2, matPanorama, (Htr*HomoMatrix), matPanorama.size(), INTER_CUBIC, BORDER_TRANSPARENT, 0);
	
	return matPanorama;

	}


Mat surf_feature_matching(Mat img1, Mat img2, int threshold)
{
	if (img1.empty() || img2.empty())
	{
		std::cout << "image load fail" << std::endl;
	}

	vector< KeyPoint > vtKeypoint_img1, vtKeypoint_img2;
	Mat img1_g, img2_g;
	Mat img1_descriptors, img2_descriptors;

	cvtColor(img1, img1_g, CV_RGB2GRAY);
	cvtColor(img2, img2_g, CV_RGB2GRAY);

	if (!img1.data || !img2.data)
	{
		std::cout << "Gray fail" << std::endl;
	}

	//SURF 기법을 이용한 특징점 검출
	Ptr<SurfFeatureDetector> Detector = SURF::create(threshold);
	Detector->detect(img1_g, vtKeypoint_img1);
	Detector->detect(img2_g, vtKeypoint_img2);
	Ptr<SurfFeatureDetector> Extractor = SURF::create();
	Extractor->compute(img1_g, vtKeypoint_img1, img1_descriptors);
	Extractor->compute(img2_g, vtKeypoint_img2, img2_descriptors);

	int constant_K = 1;
	vector<DMatch> good_matches;

	do
	{
		good_matches.clear();
		constant_K++;

		vector< DMatch > matches;
		FlannBasedMatcher Matcher;
		Matcher.match(img1_descriptors, img2_descriptors, matches);

		/*
		//BFMatch 사용
		//BFMatcher matcher(4);
		//matcher.match(matDescriptorsObject, matDescriptorsScene, matches);
		*/

		/*
		Mat matallMatches;
		//현재까지 과정을 imshow로 나타냄
		//drawMatches(img1_g, vtKeypoint_img1, img2_g, vtKeypoint_img2, matches, matallMatches, Scalar::all(-1), Scalar(-1), vector<char>(), DrawMatchesFlags::DEFAULT);
		//imshow("all-matches", matallMatches);
		*/


		double dMaxDist = 0;
		double dMinDist = 100;
		double dDistance;

		//좋은 매칭 걸러내기 
		for (int i = 0; i < img1_descriptors.rows; i++)
		{
			dDistance = matches[i].distance;

			if (dDistance < dMinDist) dMinDist = dDistance;
			if (dDistance > dMaxDist) dMaxDist = dDistance;

		}
		std::cout << "Max :" << dMaxDist << endl;
		std::cout << "Min :" << dMinDist << endl;

		for (int i = 0; i < img1_descriptors.rows; i++)
		{
			if (matches[i].distance < constant_K * dMinDist)
			{
				good_matches.push_back(matches[i]);
			}
		}

		if (constant_K > 7)
		{
			std::cout << "feature_N 값을 변경하세요" << endl;
			break;
		}
	} while (good_matches.size() < 20);


	//good-matches를 나타냄
	//Mat matGoodMatches;
	//drawMatches(img1_g, vtKeypoint_img1, img2_g, vtKeypoint_img2, good_matches, matGoodMatches, Scalar::all(-1), Scalar(-1), vector<char>(), DrawMatchesFlags::DEFAULT);
	//imshow("good-matches", matGoodMatches);
	//waitKey(200);

	std::cout << "Good Match : " << good_matches.size() << std::endl;
	std::cout << "constant K : " << constant_K << std::endl;

	vector<Point2f> img1_pt;
	vector<Point2f> img2_pt;

	for (int i = 0; i < good_matches.size(); i++)
	{
		img1_pt.push_back(vtKeypoint_img1[good_matches[i].queryIdx].pt);
		img2_pt.push_back(vtKeypoint_img2[good_matches[i].trainIdx].pt);
	}

	Mat HomoMatrix = findHomography(img2_pt, img1_pt, CV_RANSAC, 5);

	std::cout << HomoMatrix << endl;

	Mat matResult;
	Mat matPanorama;

	// 4개의 코너 구하기
	vector<Point2f> conerPt;

	conerPt.push_back(Point2f(0, 0));
	conerPt.push_back(Point2f(img1.size().width, 0));
	conerPt.push_back(Point2f(0, img1.size().height));
	conerPt.push_back(Point2f(img1.size().width, img1.size().height));

	Mat P_Trans_conerPt;
	perspectiveTransform(Mat(conerPt), P_Trans_conerPt, HomoMatrix);

	// 이미지의 모서리 계산
	double min_x, min_y, max_x, max_y;
	float min_x1, min_x2, min_y1, min_y2, max_x1, max_x2, max_y1, max_y2;

	min_x1 = min(P_Trans_conerPt.at<Point2f>(0).x, P_Trans_conerPt.at<Point2f>(1).x);
	min_x2 = min(P_Trans_conerPt.at<Point2f>(2).x, P_Trans_conerPt.at<Point2f>(3).x);
	min_y1 = min(P_Trans_conerPt.at<Point2f>(0).y, P_Trans_conerPt.at<Point2f>(1).y);
	min_y2 = min(P_Trans_conerPt.at<Point2f>(2).y, P_Trans_conerPt.at<Point2f>(3).y);
	max_x1 = max(P_Trans_conerPt.at<Point2f>(0).x, P_Trans_conerPt.at<Point2f>(1).x);
	max_x2 = max(P_Trans_conerPt.at<Point2f>(2).x, P_Trans_conerPt.at<Point2f>(3).x);
	max_y1 = max(P_Trans_conerPt.at<Point2f>(0).y, P_Trans_conerPt.at<Point2f>(1).y);
	max_y2 = max(P_Trans_conerPt.at<Point2f>(2).y, P_Trans_conerPt.at<Point2f>(3).y);
	min_x = min(min_x1, min_x2);
	min_y = min(min_y1, min_y2);
	max_x = max(max_x1, max_x2);
	max_y = max(max_y1, max_y2);

	// Transformation matrix
	Mat Htr = Mat::eye(3, 3, CV_64F);
	if (min_x < 0) {
		max_x = img2.size().width - min_x;
		Htr.at<double>(0, 2) = -min_x;
	}
	if (min_y < 0) {
		max_y = img2.size().height - min_y;
		Htr.at<double>(1, 2) = -min_y;
	}

	// 파노라마 만들기
	matPanorama = Mat(Size(max_x, max_y), CV_32F);
	warpPerspective(img1, matPanorama, Htr, matPanorama.size(), INTER_CUBIC, BORDER_CONSTANT, 0);
	warpPerspective(img2, matPanorama, (Htr*HomoMatrix), matPanorama.size(), INTER_CUBIC, BORDER_TRANSPARENT, 0);


	return matPanorama;
}



 bool checkInteriorExterior(const cv::Mat &mask, const cv::Rect &croppingMask,
	 int &top, int &bottom, int &left, int &right)
 {
	 // Return true if the rectangle is fine as it is
	 bool result = true;

	 cv::Mat sub = mask(croppingMask);
	 int x = 0;
	 int y = 0;

	 // Count how many exterior pixels are, and choose that side for
	 // reduction where mose exterior pixels occurred (that's the heuristic)

	 int top_row = 0;
	 int bottom_row = 0;
	 int left_column = 0;
	 int right_column = 0;

	 for (y = 0, x = 0; x < sub.cols; ++x)
	 {
		 // If there is an exterior part in the interior we have
		 // to move the top side of the rect a bit to the bottom
		 if (sub.at<char>(y, x) == 0)
		 {
			 result = false;
			 ++top_row;
		 }
	 }

	 for (y = (sub.rows - 1), x = 0; x < sub.cols; ++x)
	 {
		 // If there is an exterior part in the interior we have
		 // to move the bottom side of the rect a bit to the top
		 if (sub.at<char>(y, x) == 0)
		 {
			 result = false;
			 ++bottom_row;
		 }
	 }

	 for (y = 0, x = 0; y < sub.rows; ++y)
	 {
		 // If there is an exterior part in the interior
		 if (sub.at<char>(y, x) == 0)
		 {
			 result = false;
			 ++left_column;
		 }
	 }

	 for (x = (sub.cols - 1), y = 0; y < sub.rows; ++y)
	 {
		 // If there is an exterior part in the interior
		 if (sub.at<char>(y, x) == 0)
		 {
			 result = false;
			 ++right_column;
		 }
	 }

	 // The idea is to set `top = 1` if it's better to reduce
	 // the rect at the top than anywhere else.
	 if (top_row > bottom_row)
	 {
		 if (top_row > left_column)
		 {
			 if (top_row > right_column)
			 {
				 top = 1;
			 }
		 }
	 }
	 else if (bottom_row > left_column)
	 {
		 if (bottom_row > right_column)
		 {
			 bottom = 1;
		 }
	 }

	 if (left_column >= right_column)
	 {
		 if (left_column >= bottom_row)
		 {
			 if (left_column >= top_row)
			 {
				 left = 1;
			 }
		 }
	 }
	 else if (right_column >= top_row)
	 {
		 if (right_column >= bottom_row)
		 {
			 right = 1;
		 }
	 }

	 return result;
 }

 bool compareX(cv::Point a, cv::Point b)
 {
	 return a.x < b.x;
 }

 bool compareY(cv::Point a, cv::Point b)
 {
	 return a.y < b.y;
 }

 Mat crop(cv::Mat &source)
 {
	 cv::Mat gray;
	 source.convertTo(source, CV_8U);
	 cvtColor(source, gray, cv::COLOR_RGB2GRAY);

	 // Extract all the black background (and some interior parts maybe)

	 cv::Mat mask = gray > 0;

	 // now extract the outer contour
	 std::vector<std::vector<cv::Point> > contours;
	 std::vector<cv::Vec4i> hierarchy;

	 cv::findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cv::Point(0, 0));
	 cv::Mat contourImage = cv::Mat::zeros(source.size(), CV_8UC3);;

	 // Find contour with max elements

	 int maxSize = 0;
	 int id = 0;

	 for (int i = 0; i < contours.size(); ++i)
	 {
		 if (contours.at((unsigned long)i).size() > maxSize)
		 {
			 maxSize = (int)contours.at((unsigned long)i).size();
			 id = i;
		 }
	 }

	 // Draw filled contour to obtain a mask with interior parts

	 cv::Mat contourMask = cv::Mat::zeros(source.size(), CV_8UC1);
	 drawContours(contourMask, contours, id, cv::Scalar(255), -1, 8, hierarchy, 0, cv::Point());

	 // Sort contour in x/y directions to easily find min/max and next

	 std::vector<cv::Point> cSortedX = contours.at((unsigned long)id);
	 std::sort(cSortedX.begin(), cSortedX.end(), compareX);
	 std::vector<cv::Point> cSortedY = contours.at((unsigned long)id);
	 std::sort(cSortedY.begin(), cSortedY.end(), compareY);

	 int minXId = 0;
	 int maxXId = (int)(cSortedX.size() - 1);
	 int minYId = 0;
	 int maxYId = (int)(cSortedY.size() - 1);

	 cv::Rect croppingMask;

	 while ((minXId < maxXId) && (minYId < maxYId))
	 {
		 cv::Point min(cSortedX[minXId].x, cSortedY[minYId].y);
		 cv::Point max(cSortedX[maxXId].x, cSortedY[maxYId].y);
		 croppingMask = cv::Rect(min.x, min.y, max.x - min.x, max.y - min.y);

		 // Out-codes: if one of them is set, the rectangle size has to be reduced at that border

		 int ocTop = 0;
		 int ocBottom = 0;
		 int ocLeft = 0;
		 int ocRight = 0;

		 bool finished = checkInteriorExterior(contourMask, croppingMask, ocTop, ocBottom, ocLeft, ocRight);

		 if (finished == true)
		 {
			 break;
		 }

		 // Reduce rectangle at border if necessary

		 if (ocLeft)
		 {
			 ++minXId;
		 }
		 if (ocRight)
		 {
			 --maxXId;
		 }
		 if (ocTop)
		 {
			 ++minYId;
		 }
		 if (ocBottom)
		 {
			 --maxYId;
		 }
	 }

	 // Crop image with created mask

	 source = source(croppingMask);

	 return source;
 }

int main()
{
	//사진 불러오기&체크
	Mat panorama_1 = imread("panorama_1.jpg", IMREAD_COLOR);
	Mat panorama_2 = imread("panorama_2.jpg", IMREAD_COLOR);
	Mat panorama_3 = imread("panorama_3.jpg", IMREAD_COLOR);
	Mat panorama_4 = imread("panorama_4.jpg", IMREAD_COLOR);
	Mat panorama_5 = imread("panorama_5.jpg", IMREAD_COLOR);
	Mat panorama_6 = imread("panorama_6.jpg", IMREAD_COLOR);
	Mat panorama_7 = imread("panorama_7.jpg", IMREAD_COLOR);
	Mat panorama_8 = imread("panorama_8.jpg", IMREAD_COLOR);


	std::cout << "################ matches ################" << std::endl;
	find_matches(panorama_1, panorama_2, 500);
	find_matches(panorama_1, panorama_3, 500);
	find_matches(panorama_1, panorama_4, 500);
	find_matches(panorama_1, panorama_5, 500);
	find_matches(panorama_1, panorama_6, 500);
	find_matches(panorama_1, panorama_7, 500);
	find_matches(panorama_1, panorama_8, 500);


	
	//up
	Mat sum_12 = sift_panorama(panorama_2, panorama_1, 500);
	sum_12 = crop(sum_12);
	Mat sum_23 = sift_panorama(panorama_2, panorama_3, 500);
	sum_23 = crop(sum_23);
	Mat sum_123 = sift_panorama(sum_12, sum_23, 500);
	sum_123 = crop(sum_123);
	std::cout << "################ up ################" << std::endl;
	//down
	Mat sum_67 = sift_panorama(panorama_6, panorama_7, 500);
	sum_67 = crop(sum_67);
	Mat sum_56 = sift_panorama(panorama_6, panorama_5, 500);
	sum_56 = crop(sum_56);
	Mat sum_567 = sift_panorama(sum_56, sum_67, 500);
	sum_567 = crop(sum_567);
	std::cout << "################ down ################" << std::endl;

	Mat sum_123567 = sift_panorama(sum_123, sum_567, 500);
	
	//namedWindow("result1", WINDOW_FREERATIO);
	//imshow("result1", sum_123);

	//namedWindow("result2", WINDOW_FREERATIO);
	//imshow("result2", sum_567);

	namedWindow("result3", WINDOW_FREERATIO);
	imshow("result3", sum_123567);
	
	Mat panorama_t_1 = imread("assignment3_data/hill/1.jpg", IMREAD_COLOR);
	Mat panorama_t_2 = imread("assignment3_data/hill/2.jpg", IMREAD_COLOR);
	Mat panorama_t_3 = imread("assignment3_data/hill/3.jpg", IMREAD_COLOR);

	Mat sum_t_12 = sift_panorama(panorama_t_1, panorama_t_2, 500);
	sum_t_12 = crop(sum_t_12);
	Mat sum_t_23 = sift_panorama(panorama_t_2, panorama_t_3, 500);
	sum_t_23 = crop(sum_t_23);
	Mat sum_t_123 = sift_panorama(sum_t_12, sum_t_23, 500);
	sum_t_123 = crop(sum_t_123);

	namedWindow("result_t", WINDOW_FREERATIO);
	imshow("result_t", sum_t_123);

	waitKey(0);
}



