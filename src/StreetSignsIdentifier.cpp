#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <exception>

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
StreetSignsIdentifier::StreetSignsIdentifier(std::string warningSignsClassifierPath,std::string speedLimitSignsClassifierPath,std::string noParkingSignsClassifierPath)
{
	if(warningSignsClassifierPath=="" || !warningSignsClassifier.load(warningSignsClassifierPath))
		THROW_NICE(invalid_argument,"couldn't load warning signs classifier at \""+warningSignsClassifierPath+"\"");
}

/**
 * Identifies the street signs in the image
 * @param  img the image to be analyzed
 * @return     a vector containing all the identified signs
 */
std::vector<StreetSign> StreetSignsIdentifier::identify(cv::Mat img)
{
	Mat reds = findReds(img);
  displayImage(reds,"findReds",1000);
  vector<Mat> rgb_planes;
  split(img, rgb_planes);
  reds = reds/255;
  rgb_planes[0] = rgb_planes[0].mul(reds);
  rgb_planes[1] = rgb_planes[1].mul(reds);
  rgb_planes[2] = rgb_planes[2].mul(reds);
  Mat imgP;
  merge(rgb_planes, imgP);
  displayImage(imgP,"preprocessed image",1000);
  Mat imgP_gray;
  cvtColor(imgP, imgP_gray, CV_BGR2GRAY);

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
  edgeImg = edgeImg.mul(reds);
  displayImage(edgeImg,"masked edges",1000);


  std::vector<Vec3f> circles;
  Mat imgCirc;
  img.copyTo(imgCirc);
  getCircles(imgP_gray, circles);
  drawCirclesOnImg(circles, imgCirc, 15);
  displayImage(imgCirc,"circles with preproc",1000);

/*
  circles.clear();
  img.copyTo(imgCirc);
  getCircles(imgGray, circles);
  drawCirclesOnImg(circles, imgCirc, 15);
  displayImage(imgCirc,"circles with gray",1000);*/

  vector<StreetSign> ret;
	detectAllSigns(img, ret);
  return ret;
}

/**
 * [StreetSignsIdentifier::findReds description]
 * @param  img [description]
 * @return     [description]
 */
cv::Mat StreetSignsIdentifier::findReds(const cv::Mat& img)
{
  Mat imgR1 = findRedsV1(img);
  Mat imgR2 = findRedsV2(img);

  displayImage(imgR1,"findRedsV1",1000);
  displayImage(imgR2,"findRedsV2",1000);
  resize(imgR1,imgR1,img.size(),0,0,INTER_LINEAR);
  resize(imgR2,imgR2,img.size(),0,0,INTER_LINEAR);
  return imgR1.mul(imgR2);//intersect them
}

/**
 * trises to identify the red pixels in the image, it gets combined with \ref findRedsV1
 * @param img input image
 * @return a binary image with 1 in correspondence to the red pixels of img and 0 elsewhere
 */
cv::Mat StreetSignsIdentifier::findRedsV1(const cv::Mat& img)
{
  Mat imgResized;
  resize(img,imgResized,Size(),0.25,0.25);

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  Mat blurredImg;
  GaussianBlur(imgResized,blurredImg, Size(5,5), 1);


  unsigned long blurTime =  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count();
  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"findRedsV1: blur took: "<<blurTime<<"us"<<endl;
  //find the red pixel by analyzing the hue


  Mat imgHls;
  cvtColor(blurredImg,imgHls,COLOR_BGR2HLS);
  vector<cv::Mat> hls_planes;
  split( imgHls, hls_planes );

  unsigned long hsvTime =  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count() - blurTime;
  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"findRedsV1: hsv conversion took: "<<hsvTime<<"us"<<endl;

  Mat redDist = hls_planes[0];//we will transform the hue in the "distance to the red hue"
  //equalizeHist(hsv_planes[0],redDist);//before calculating the distances equalize the hue
  /*
  for(MatIterator_<uchar>  it = redDist.begin<uchar>(), end = redDist.end<uchar>();
     it != end; ++it)
  {
    //int prev = *it;
    if(*it>179/2.0)
      *it = (8*(*it))-179*7;
    else
      *it = (-8*(*it))+179;
  }*/
  {
    Mat t;
    threshold(hls_planes[0], t, 15, 255, CV_THRESH_BINARY_INV);
    Mat t2;
    threshold(hls_planes[0], t2, 165, 255, CV_THRESH_BINARY);
    bitwise_or(t,t2,hls_planes[0]);
  }



  unsigned long redDistTime =  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count() - hsvTime;
  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"findRedsV1: red Dist computation took: "<<redDistTime<<"us"<<endl;

  //multiply the distances to the red by the saturation, to enhace the pixels that are really red
  Mat redDistBySat = redDist.mul(hls_planes[2])*1/256.0;

  //threshold what we obtained

  Mat redDistBySatThresh;
  double min=0,max=255;//ignored if using otsu
  //minMaxLoc(redDistBySat, &min, &max);
  threshold(redDistBySat, redDistBySatThresh, 0.4*min+0.6*max, 255, THRESH_BINARY | THRESH_OTSU);

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

  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"findRedsV1 took "<<std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<<" us"<<endl;
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
    displayImage(redDistBySatThresh, "redDistBySatThresh", 1000);
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

  return redDistBySatThresh;
}

/**
 * trises to identify the red pixels in the image, it gets combined with \ref findRedsV1
 * @param img input image
 * @return a binary image with 1 in correspondence to the red pixels of img and 0 elsewhere
 */
cv::Mat StreetSignsIdentifier::findRedsV2(const cv::Mat& img)
{
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  Mat imgResized;
  resize(img,imgResized,Size(),0.25,0.25);

  Mat blurredImg;
  GaussianBlur(imgResized,blurredImg, Size(5,5), 1);

  std::chrono::steady_clock::time_point blurredTime = std::chrono::steady_clock::now();
  //equalize the lightness channel
  Mat imgLab;
  cvtColor(blurredImg,imgLab,COLOR_BGR2Lab);

  std::chrono::steady_clock::time_point toLabTime = std::chrono::steady_clock::now();

  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"findRedsV2: lab conversion took: "<<std::chrono::duration_cast<std::chrono::microseconds>(toLabTime - blurredTime).count()<<"us"<<endl;


  vector<cv::Mat> lab_planes;
  split( imgLab, lab_planes );
  Mat eqL;
  equalizeHist(lab_planes[0],eqL);
  lab_planes[0]=eqL;
  /*
  merge(lab_planes, imgLab);
  Mat blurredImgEqL;
  cvtColor(imgLab,blurredImgEqL,COLOR_Lab2BGR);
  displayImage(blurredImgEqL,"blurred eql",1000);
  cvtColor(blurredImgEqL,imgLab,COLOR_BGR2Lab);
  displayImage(imgLab,"imgLab eql",1000);*/


  Mat imgLabF;
  imgLab.convertTo(imgLabF, CV_32FC3, 1.0/255);

  Vec3f red(0.5,1,1);

  std::chrono::steady_clock::time_point beforeDistsComputation = std::chrono::steady_clock::now();

  for(MatIterator_<Vec3f>  it = imgLabF.begin<Vec3f>(), end = imgLabF.end<Vec3f>();
     it != end; ++it)
  {
    //cout<<"img = "<<(*it)<<" dist = "<<norm(*it-red)<<endl;
    *it = Vec3f(norm(*it-red),0,0);
  }

  std::chrono::steady_clock::time_point afterDistsComputation = std::chrono::steady_clock::now();
  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"findRedsV2: dists computation took: "<<std::chrono::duration_cast<std::chrono::microseconds>(afterDistsComputation - beforeDistsComputation).count()<<"us"<<endl;


  split(imgLabF, lab_planes);
  Mat res;
  lab_planes[0].convertTo(res, CV_8UC3, 255/1.73);
/*  Mat resEq;
  equalizeHist(res, resEq);
  displayImage(resEq,"resEq",1000);*/
  Mat resThresh;
  double min=0,max=255;//ignored if using otsu
  minMaxLoc(res, &min, &max);
  threshold(res, resThresh, min+(max-min)*0.35, 255, THRESH_BINARY_INV);

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"findRedsV2 took: "<<std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<<"us"<<endl;


  if(verbosity>=VERBOSITY_SHOW_IMAGES)
  {
    displayImage(res,"res",1000);
    displayImage(imgLab,"imgLab",1000);
  }


  return resThresh;
}

void StreetSignsIdentifier::getEdges(const cv::Mat& inputImg, cv::Mat& edgeImg)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  	Canny(inputImg,edgeImg,100,400);
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    unsigned long edgeDetectionTimeMicro =  std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    if(verbosity>=VERBOSITY_TEXT_ONLY)
      cout<<"edge detection required "<<edgeDetectionTimeMicro<<" us"<<endl;
}


void StreetSignsIdentifier::getCircles(const cv::Mat& inputImg, std::vector<cv::Vec3f>& circles)
{
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  HoughCircles(inputImg, circles, HOUGH_GRADIENT,4, inputImg.cols/30, 200, 25, inputImg.cols/50, inputImg.cols*0.75 );
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
		THROW_NICE(invalid_argument,"unsupported image (has "+std::to_string(circlesImage.channels())+" channels)");
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


/**
 * Detects objects in the provided image using a cacade classifier
 * @param inputImage the image to serch into
 * @param classifier the classifier to use
 * @param detections here are returned the detected objects locations
 */
void StreetSignsIdentifier::detectWithCascade(cv::Mat& inputImage, cv::CascadeClassifier& classifier, std::vector<Rect>& detections)
{
	Mat convertedImage;
	if(inputImage.type()==CV_8UC3)
		cvtColor(inputImage, convertedImage, COLOR_BGR2GRAY);
	else if(inputImage.type()==CV_8U)
		convertedImage = inputImage;
	else
		THROW_NICE(invalid_argument,"provided image has unsupported format, only CV_8U and CV_8UC3 are supported");

	float minSize = inputImage.cols/30;
	float maxSize = inputImage.cols/2;
  classifier.detectMultiScale( convertedImage, detections,1.1,3,0,Size2f(minSize,minSize),Size2f(maxSize,maxSize));
}

void StreetSignsIdentifier::detectWarningSigns(cv::Mat& inputImage, std::vector<StreetSign_Warning>& detectedSigns)
{
	vector<Rect> detections;
	detectWithCascade(inputImage,warningSignsClassifier, detections);

	//build the descritpions
	for(Rect r : detections)
		detectedSigns.push_back(StreetSign_Warning(Point2f(r.x+r.width/2,r.y+r.height/2),r.size()));
}

void StreetSignsIdentifier::detectAllSigns(cv::Mat& inputImage, std::vector<StreetSign>& detectedSigns)
{
	vector<StreetSign_Warning> warnings;
	detectWarningSigns(inputImage, warnings);
	for(StreetSign_Warning sw : warnings)
		detectedSigns.push_back(sw);
	//detectedSigns.insert(detectedSigns.end(), warnings.begin(), warnings.end());
/*	vector<StreetSign_Warning> speeds;
	detectSpeedLimitSigns(inputImage, speeds);
	detectedSigns.insert(detectedSigns.end(), speeds.begin(), speeds.end());
	vector<StreetSign_Warning> noParks;
	detectNoParkingSigns(inputImage, noParks);
	detectedSigns.insert(detectedSigns.end(), noParks.begin(), noParks.end());*/
}
