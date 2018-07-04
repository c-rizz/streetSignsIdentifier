

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;
/**
 * Displays an image
 * @param image     the image to be displayed
 * @param winName   the name of the window to create
 * @param winWidth  the windows width
 * @param winHeight the window height
 */
void displayImage(cv::Mat& image, std::string winName, int winWidth, int winHeight=-1)
{
	namedWindow(winName, cv::WINDOW_NORMAL);
	//cout<<"image is "<<img.cols<<"x"<<img.rows<<endl;
	if(winHeight<0)
	 winHeight = (int)((double)winWidth/image.cols*image.rows);
	//cout<<"resizing window to "<<winWidth<<"x"<<winHeight<<endl;
	resizeWindow(winName,winWidth,winHeight);
	imshow(winName,image);

	cv::waitKey(0);
	destroyWindow(winName);
}
