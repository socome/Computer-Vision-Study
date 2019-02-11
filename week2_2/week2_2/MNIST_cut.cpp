#include <opencv/cv.h>

#include <opencv/highgui.h>



using namespace cv;

using namespace std;

/*

int main() {

	int cntr_0 = 0;
	int cntr_1 = 0;
	int cntr_2 = 0;
	int cntr_3 = 0;
	int cntr_4 = 0;
	int cntr_5 = 0;
	int cntr_6 = 0;
	int cntr_7 = 0;
	int cntr_8 = 0;
	int cntr_9 = 0;

	std::string savingName;
	// 이미지 불러오기 (read image).
	Mat image = imread("MNIST_IMAGE.PNG");

	// 에러 처리 (error).
	if (!image.data) {

		return -1;

	}


	for (int i = 1; i < 51; i++)
	{
		for (int j = 1; j < 101; j++)
		{
			// 관심영역 설정 (set ROI (X, Y, W, H)).
			Rect rect((j - 1) * 20,(i-1)*20, 20, 20);
			// 관심영역 자르기 (Crop ROI).
			Mat subImage = image(rect);
			if (i <= 5)
			{
				// save
			savingName = "D:/Computer-Vision-Study/week2_2/week2_2/MNIST/data 0_" + std::to_string(++cntr_0) + ".PNG";
			}
			else if (i <= 10)
			{
				// save
				savingName = "D:/Computer-Vision-Study/week2_2/week2_2/MNIST/data 1_" + std::to_string(++cntr_1) + ".PNG";
			}
			else if (i <= 15)
			{
				// save
				savingName = "D:/Computer-Vision-Study/week2_2/week2_2/MNIST/data 2_" + std::to_string(++cntr_2) + ".PNG";
			}
			else if (i <= 20)
			{
				// save
				savingName = "D:/Computer-Vision-Study/week2_2/week2_2/MNIST/data 3_" + std::to_string(++cntr_3) + ".PNG";
			}
			else if (i <= 25)
			{
				// save
				savingName = "D:/Computer-Vision-Study/week2_2/week2_2/MNIST/data 4_" + std::to_string(++cntr_4) + ".PNG";
			}
			else if (i <= 30)
			{
				// save
				savingName = "D:/Computer-Vision-Study/week2_2/week2_2/MNIST/data 5_" + std::to_string(++cntr_5) + ".PNG";
			}
			else if (i <= 35)
			{
				// save
				savingName = "D:/Computer-Vision-Study/week2_2/week2_2/MNIST/data 6_" + std::to_string(++cntr_6) + ".PNG";
			}
			else if (i <= 40)
			{
				// save
				savingName = "D:/Computer-Vision-Study/week2_2/week2_2/MNIST/data 7_" + std::to_string(++cntr_7) + ".PNG";
			}
			else if (i <= 45)
			{
				// save
				savingName = "D:/Computer-Vision-Study/week2_2/week2_2/MNIST/data 8_" + std::to_string(++cntr_8) + ".PNG";
			}
			else if (i <= 50)
			{
				// save
				savingName = "D:/Computer-Vision-Study/week2_2/week2_2/MNIST/data 9_" + std::to_string(++cntr_9) + ".PNG";
			}

			
			
			imwrite(savingName, subImage);

		}
	}

	


	return 0;

}
*/