#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "StreetSignsIdentifier.hpp"
#include "utils.hpp"


using namespace std;
using namespace cv;


const int StreetSignsIdentifier::VERBOSITY_NORMAL=0;
const int StreetSignsIdentifier::VERBOSITY_TEXT_ONLY=1;
const int StreetSignsIdentifier::VERBOSITY_SHOW_IMAGES=2;

/**
 * Initializes the StreetSignsIdentifier
 */
StreetSignsIdentifier::StreetSignsIdentifier()
{
  //nothing to do by now
}

/**
 * Identifies the street signs in the image
 * @param  img the image to be analyzed
 * @return     a vector containing all the identified signs
 */
std::vector<StreetSign> StreetSignsIdentifier::identify(cv::Mat img)
{
  Mat imgP;
  img.copyTo(imgP);
	preprocessForRedSigns(imgP);

  displayImage(imgP,"preprocessed image",1000);

  Mat imgGray;
  Mat imgGrayNeq;
  cvtColor(img, imgGrayNeq, CV_BGR2GRAY);
  equalizeHist(imgGrayNeq, imgGray);

  displayImage(imgGray,"grayscale",1000);

  Mat edgeImg;
  getEdges(imgP, edgeImg);
  displayImage(edgeImg,"edges with preproc",1000);
  getEdges(imgGray, edgeImg);
  displayImage(edgeImg,"edges with grayscale",1000);


  std::vector<Vec3f> circles;
  Mat imgCirc;
  img.copyTo(imgCirc);
  getCircles(imgP, circles);
  drawCirclesOnImg(circles, imgCirc, 10000000);
  displayImage(imgCirc,"circles with preproc",1000);

  circles.clear();
  img.copyTo(imgCirc);
  getCircles(imgGray, circles);
  drawCirclesOnImg(circles, imgCirc, 10000000);
  displayImage(imgCirc,"circles with gray",1000);

  vector<StreetSign> ret;
  return ret;
}


/**
 * Preprocesses the image to facilitate the red signs detection
 * @param img [description]
 */
void StreetSignsIdentifier::preprocessForRedSigns(cv::Mat& img)
{
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  Mat blurredImg;
  GaussianBlur(img,blurredImg, Size(5,5), 1);

  //equalize the lightness channel
  Mat imgHls;
  cvtColor(blurredImg,imgHls,COLOR_BGR2HLS);
  vector<cv::Mat> hls_planes;
  split( imgHls, hls_planes );
  Mat eqL;
  equalizeHist(hls_planes[1],eqL);


  unsigned long eqLTime =  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count();
  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"proproc: lightness equalization took: "<<eqLTime<<"us"<<endl;


  unsigned long blurTime =  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count();
  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"proproc: blur took: "<<blurTime<<"us"<<endl;
  //find the red pixel by analyzing the hue


  unsigned long hsvTime =  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count() - blurTime;
  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"proproc: hsv conversion took: "<<hsvTime<<"us"<<endl;

  Mat redDist = hls_planes[0];//we will transform the hue in the "distance to the red hue"
  //equalizeHist(hsv_planes[0],redDist);//before calculating the distances equalize the hue
  for(MatIterator_<uchar>  it = redDist.begin<uchar>(), end = redDist.end<uchar>();
     it != end; ++it)
  {
    //int prev = *it;
    if(*it>179/2.0)
      *it = (8*(*it))-179*7;
    else
      *it = (-8*(*it))+179;
  /*  if(prev==0)
      cout<<prev<<" --> "<<((int)(*it))<<endl;*/
    //*it = (((*it)>128)? (2*(*it)-256) : (-2*(*it)+256));
  }


  unsigned long redDistTime =  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count() - hsvTime;
  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"proproc: red Dist computation took: "<<redDistTime<<"us"<<endl;

  //multiply the distances to the red by the saturation, to enhace the pixels that are really red
  Mat redDistBySat = redDist.mul(hls_planes[2])*1/256.0;

  Mat redDistBySatByLightness = redDistBySat.mul(eqL)*1/256.0;
  //threshold what we obtained
/*
  Mat redDistBySatThresh;
  double min=0,max=255;//ignored if using otsu
  //minMaxLoc(redDistBySat, &min, &max);
  threshold(redDistBySat, redDistBySatThresh, 0.4*min+0.6*max, 255, THRESH_BINARY | THRESH_OTSU);
*/

  //and that's the result
  img = redDistBySat;//ByLightness;
/*
  //another idea is to multiply the red bgr channel by the saturation
  vector<cv::Mat> bgr_planes;
  split( blurredImgEqL, bgr_planes );
  Mat redBySat = bgr_planes[2].mul(hsv_planes[1])*1/256.0;
  //again, we threshold
  Mat redBySatThresh;
  minMaxLoc(redDistBySat, &min, &max);
  threshold(redBySat, redBySatThresh, 0.4*min+0.6*max, 255, THRESH_BINARY);

  //multiply the two ideas, so to keep their
  img = redDistBySatThresh.mul(redBySatThresh);
*/
  std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();

  preprocessForRedDurationMicro =  std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"preprocessForRedSigns required "<<preprocessForRedDurationMicro<<" us"<<endl;
  //if you want to see the intermediate images
  if(verbosity>=VERBOSITY_SHOW_IMAGES)
  {
    displayImage(blurredImg,"blurred img",1000);
    int c = 0;
    for(Mat plane : hls_planes)
    {
      string imgName = "HlsPlane"+to_string(c++);
      displayImage(plane,imgName,1000);
    }
    displayImage(redDist,"redDist",1000);
    displayImage(redDistBySat, "redDistBySat", 1000);
    //displayImage(redDistBySatThresh, "redDistBySatThresh", 1000);
    /*c = 0;
    for(Mat plane : bgr_planes)
    {
      string imgName = "bgrPlane"+to_string(c++);
      displayImage(plane,imgName,1000);
    }
    displayImage(redBySat, "redBySat", 1000);
    displayImage(redBySatThresh, "redBySatThresh", 1000);
    */
  }
}


void StreetSignsIdentifier::getEdges(const cv::Mat& inputImg, cv::Mat& edgeImg)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  	Canny(inputImg,edgeImg,150,400);
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    unsigned long edgeDetectionTimeMicro =  std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    if(verbosity>=VERBOSITY_TEXT_ONLY)
      cout<<"edge detection required "<<edgeDetectionTimeMicro<<" us"<<endl;
}


void StreetSignsIdentifier::getCircles(const cv::Mat& inputImg, std::vector<cv::Vec3f>& circles)
{
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  HoughCircles(inputImg, circles, HOUGH_GRADIENT,2, 10, 600, 25, 0, 50 );
  std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
  unsigned long circlesDetectionTimeMicro =  std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"circles detection required "<<circlesDetectionTimeMicro<<" us"<<endl;
}

/**
 * Set the verbosity of the object
 * @param isOn set this one of either VERBOSITY_NORMAL, VERBOSITY_TEXT_ONLY or VERBOSITY_SHOW_IMAGES
 */
void StreetSignsIdentifier::setVerbosity(int level)
{
  verbosity = level;
}


/**
 * Draws the provided circles on the provied image
 * @param circles          The circles set
 * @param circlesImage     The image to draw upon
 * @param circlesToDrawNum How many circles to draw (only the first circlesToDrawNum circles in circles will be drawn)
 */
void StreetSignsIdentifier::drawCirclesOnImg(std::vector<cv::Vec3f>& circles, cv::Mat& circlesImage, int circlesToDrawNum)
{
	//If the image is grayscale I convert it to rgb
	if(circlesImage.channels()==1)
	{
		cv::Mat imgRgb;
		cv::cvtColor(circlesImage, imgRgb, cv::COLOR_GRAY2BGR);
		circlesImage = imgRgb;
	}
	else if(circlesImage.channels()!=3)
	{
		throw std::invalid_argument("unsupported image (has "+std::to_string(circlesImage.channels())+" channels)");
	}

	for(unsigned int i=0;i<min((unsigned int)circlesToDrawNum,(unsigned int)circles.size());i++)
	{
		double x = circles.at(i)[0];
		double y = circles.at(i)[1];
		double r = circles.at(i)[2];
		//cout<<"circles["<<i<<"] = "<<circles.at(i)<<endl;
		circle(circlesImage, cv::Point2f(x,y), r, Scalar(0,255,0), 3, 8, 0 );
	}
}
