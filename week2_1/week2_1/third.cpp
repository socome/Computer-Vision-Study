#include <opencv2/dpm.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>

#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace cv::dpm;
using namespace std;

static void help()
{
	cout << "\nThis is a demo of \"Deformable Part-based Model (DPM) cascade detection API\" using web camera.\n"
		"Call:\n"
		"./example_dpm_cascade_detect_camera <model_path>\n"
		<< endl;
}

void drawBoxes(Mat &frame,
	vector<DPMDetector::ObjectDetection> ds,
	Scalar color,
	string text);

int main(int argc, char** argv)
{
	const char* keys =
	{
		"{@model_path    | | Path of the DPM cascade model}"
	};

	CommandLineParser parser(argc, argv, keys);

	string model_path = "person.xml";

	if (model_path.empty())
	{
		help();
		return -1;
	}

	cv::Ptr<DPMDetector> detector = DPMDetector::create(vector<string>(1, model_path));
	
	VideoCapture test_video;

	test_video.open("vtest.avi");

	if (!test_video.isOpened())
	{
		cerr << "Fail to open default camera (0)!" << endl;
		return -1;
	}

	Mat frame;
	namedWindow("DPM Cascade Detection", 1);
	// the color of the rectangle
	Scalar color(0, 255, 255); // yellow

	while (test_video.read(frame))
	{
		vector<DPMDetector::ObjectDetection> ds;

		Mat image;
		frame.copyTo(image);

		// detection
		detector->detect(image, ds);

		// draw boxes
		string text = format("person");
		drawBoxes(frame, ds, color, text);

		imshow("DPM Cascade Detection", frame);

		if (waitKey(30) >= 0)
			break;
	}

	return 0;
}

void drawBoxes(Mat &frame,
	vector<DPMDetector::ObjectDetection> ds,
	Scalar color,
	string text)
{
	for (unsigned int i = 0; i < ds.size(); i++)
	{
		rectangle(frame, ds[i].rect, color, 2);
	}

	// draw text on image
	Scalar textColor(0, 0, 250);
	putText(frame, text, Point(10, 50), FONT_HERSHEY_PLAIN, 2, textColor, 2);
}