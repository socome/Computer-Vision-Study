
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
using namespace cv::xfeatures2d;

const int LOOP_NUM = 1;
const int GOOD_PTS_MAX = 50;
const float GOOD_PORTION = 0.15f;

int64 work_begin = 0;
int64 work_end = 0;


struct SURFDetector
{
	Ptr<Feature2D> surf;
	SURFDetector(double hessian = 800.0)
	{
		surf = SURF::create(hessian);
	}
	template<class T>
	void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
	{
		surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
	}
};

template<class KPMatcher>
struct SURFMatcher
{
	KPMatcher matcher;
	template<class T>
	void match(const T& in1, const T& in2, std::vector<cv::DMatch>& matches)
	{
		matcher.match(in1, in2, matches);
	}
};

static Mat drawGoodMatches(
	const Mat& img1,
	const Mat& img2,
	const std::vector<KeyPoint>& keypoints1,
	const std::vector<KeyPoint>& keypoints2,
	std::vector<DMatch>& matches,
	std::vector<Point2f>& scene_corners_
)
{
	//-- Sort matches and preserve top 10% matches
	std::sort(matches.begin(), matches.end());
	std::vector< DMatch > good_matches;
	double minDist = matches.front().distance;
	double maxDist = matches.back().distance;

	const int ptsPairs = std::min(GOOD_PTS_MAX, (int)(matches.size() * GOOD_PORTION));
	for (int i = 0; i < ptsPairs; i++)
	{
		good_matches.push_back(matches[i]);
	}
	std::cout << "\nMax distance: " << maxDist << std::endl;
	std::cout << "Min distance: " << minDist << std::endl;

	std::cout << "Calculating homography using " << ptsPairs << " point pairs." << std::endl;

	// drawing the results
	Mat img_matches;

	drawMatches(img1, keypoints1, img2, keypoints2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}
	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0, 0);
	obj_corners[1] = Point(img1.cols, 0);
	obj_corners[2] = Point(img1.cols, img1.rows);
	obj_corners[3] = Point(0, img1.rows);
	std::vector<Point2f> scene_corners(4);

	Mat H = findHomography(obj, scene, RANSAC);
	perspectiveTransform(obj_corners, scene_corners, H);

	scene_corners_ = scene_corners;

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches,
		scene_corners[0] + Point2f((float)img1.cols, 0), scene_corners[1] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);
	line(img_matches,
		scene_corners[1] + Point2f((float)img1.cols, 0), scene_corners[2] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);
	line(img_matches,
		scene_corners[2] + Point2f((float)img1.cols, 0), scene_corners[3] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);
	line(img_matches,
		scene_corners[3] + Point2f((float)img1.cols, 0), scene_corners[0] + Point2f((float)img1.cols, 0),
		Scalar(0, 255, 0), 2, LINE_AA);
	return img_matches;
}

////////////////////////////////////////////////////
// This program demonstrates the usage of SURF_OCL.
// use cpu findHomography interface to calculate the transformation matrix

int main(int argc, char* argv[])
{


		UMat img1;

		std::string leftName = "object1.jpg";
		imread(leftName, IMREAD_GRAYSCALE).copyTo(img1);
		if (img1.empty())
		{
			std::cout << "Couldn't load " << leftName << std::endl;
			return EXIT_FAILURE;
		}


		//declare input/output
		std::vector<KeyPoint> keypoints1, keypoints2;
		std::vector<DMatch> matches;


		UMat _descriptors1, _descriptors2;
		Mat descriptors1 = _descriptors1.getMat(ACCESS_RW),
			descriptors2 = _descriptors2.getMat(ACCESS_RW);


		//instantiate detectors/matchers
		SURFDetector surf;

		SURFMatcher<BFMatcher> matcher;

		VideoCapture cap1(0);

		if (!cap1.isOpened())
		{
			printf("첫번째 카메라를 열수 없습니다. \n");
		}

		UMat frame1;

		namedWindow("camera1", 1);

		for (;;)
		{
			cap1 >> frame1;

			imshow("camera1", frame1);

			//std::cout << "정상작동중" << std::endl;

			//-- start of timing section

			for (int i = 0; i <= LOOP_NUM; i++)
			{

				std::cout << "LOOP_NUM" << i << std::endl;
				surf(img1.getMat(ACCESS_READ), Mat(), keypoints1, descriptors1);
				surf(frame1.getMat(ACCESS_READ), Mat(), keypoints2, descriptors2);
				matcher.match(descriptors1, descriptors2, matches);
			}

			//std::cout << "FOUND " << keypoints1.size() << " keypoints on first image" << std::endl;
			//std::cout << "FOUND " << keypoints2.size() << " keypoints on second image" << std::endl;



			std::vector<Point2f> corner;
			Mat img_matches = drawGoodMatches(img1.getMat(ACCESS_READ), frame1.getMat(ACCESS_READ), keypoints1, keypoints2, matches, corner);

			//-- Show detected matches

			namedWindow("surf matches", 0);
			imshow("surf matches", img_matches);
			//	    imwrite(outpath, img_matches);


			if (waitKey(30) >= 0) break;
		}


	return EXIT_SUCCESS;
}
