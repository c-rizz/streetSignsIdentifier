#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xphoto.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <exception>

#include "StreetSignsIdentifier.hpp"
#include "utils.hpp"


using namespace std;
using namespace cv;


const int StreetSignsIdentifier::VERBOSITY_NORMAL=0;
const int StreetSignsIdentifier::VERBOSITY_TEXT_ONLY=1;
const int StreetSignsIdentifier::VERBOSITY_SHOW_IMAGES=2;
const int StreetSignsIdentifier::VERBOSITY_SHOW_MORE_IMAGES=3;

/**
 * Initializes the StreetSignsIdentifier
 */
StreetSignsIdentifier::StreetSignsIdentifier(std::string warningSignsClassifierPath,std::string speedLimitSignsClassifierPath,std::string roundSignsClassifierPath)
{
	if(warningSignsClassifierPath=="" || !warningSignsClassifier.load(warningSignsClassifierPath))
		THROW_NICE(invalid_argument,"couldn't load warning signs classifier at \""+warningSignsClassifierPath+"\"");
	if(roundSignsClassifierPath=="" || !roundSignsClassifier.load(roundSignsClassifierPath))
		THROW_NICE(invalid_argument,"couldn't load round signs classifier at \""+roundSignsClassifierPath+"\"");

	buildKnn();
}


/**
 * Identifies the street signs in the image
 * @param  img the image to be analyzed
 * @return     a vector containing all the identified signs
 */
vector<shared_ptr<StreetSign>> StreetSignsIdentifier::identify(cv::Mat& img)
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	Mat imgWb;

	std::chrono::steady_clock::time_point beforeWb = std::chrono::steady_clock::now();
	xphoto::createSimpleWB()->balanceWhite(img, imgWb);
	std::chrono::steady_clock::time_point afterWb = std::chrono::steady_clock::now();
	unsigned long wbDuration =  std::chrono::duration_cast<std::chrono::milliseconds>(afterWb - beforeWb).count();
	if(verbosity>=VERBOSITY_TEXT_ONLY)
		cout<<"white balance required "<<wbDuration<<" ms"<<endl;


	std::chrono::steady_clock::time_point beforeWbVisualization = std::chrono::steady_clock::now();
	if(verbosity>=VERBOSITY_SHOW_IMAGES)
  	displayImage(imgWb,"white balanced image",1000);
	std::chrono::steady_clock::time_point afterWbVisualization = std::chrono::steady_clock::now();
	unsigned long wbVisualizationDuration =  std::chrono::duration_cast<std::chrono::milliseconds>(afterWbVisualization - beforeWbVisualization).count();



  vector<shared_ptr<StreetSign>> detectedSigns;
	//detectAllSigns(imgWb, detectedSigns);

	vector<StreetSign_Warning> warnings;
	detectWarningSigns(imgWb, warnings);
	if(verbosity>=VERBOSITY_TEXT_ONLY)
		cout<<"detected "<<warnings.size()<<" warning signs"<<endl;
	vector<StreetSign> roundSigns;
	detectRoundRedSigns(imgWb, roundSigns);
	if(verbosity>=VERBOSITY_TEXT_ONLY)
		cout<<"detected "<<roundSigns.size()<<" round signs"<<endl;


	for(StreetSign_Warning sw : warnings)
		detectedSigns.push_back(std::make_shared<StreetSign_Warning>(sw));
	for(StreetSign rs : roundSigns)
		detectedSigns.push_back(std::make_shared<StreetSign>(rs));

	unsigned long redsVisualizationDuration =  0;
	if(detectedSigns.size()>0)
	{
		Mat reds = findReds(imgWb);
		std::chrono::steady_clock::time_point beforeRedsVisualization = std::chrono::steady_clock::now();
		if(verbosity>=VERBOSITY_SHOW_IMAGES)
  		displayImage(reds,"findReds",1000);
		std::chrono::steady_clock::time_point afterRedsVisualization = std::chrono::steady_clock::now();
		redsVisualizationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(afterRedsVisualization - beforeRedsVisualization).count();
		filterDetectionsByBinaryMask(detectedSigns,reds);
		filterSignsContainedInBiggerOnes(detectedSigns);
	}

	for(shared_ptr<StreetSign>& ssp: detectedSigns)
	{
		if(ssp->getType()==StreetSign::SignType::GENERIC)
			classifyRoundRedSign(imgWb, ssp);//classify the signs
	}

	//remove the non classified signs
	for(unsigned int i=0;i<detectedSigns.size();i++)
	{
		shared_ptr<StreetSign> ssp = detectedSigns.at(i);
		if(ssp->getType()==StreetSign::SignType::GENERIC)
		{
			detectedSigns.erase(detectedSigns.begin()+i);
			i--;
		}
	}

	if(verbosity>=VERBOSITY_TEXT_ONLY)
	{
		int warningsCount = 0;
		int noParkCount = 0;
		int speedLimitCount = 0;
		int genericCount = 0;
		for(shared_ptr<StreetSign> ssp: detectedSigns)
		{
			if(ssp->getType()==StreetSign::SignType::WARNING)
				warningsCount++;
			else if(ssp->getType()==StreetSign::SignType::NO_PARKING)
				noParkCount++;
			else if(ssp->getType()==StreetSign::SignType::SPEED)
				speedLimitCount++;
			else
				genericCount++;
		}

		cout<<"warnings = "<<warningsCount<<endl;
		cout<<"noParkCount = "<<noParkCount<<endl;
		cout<<"speedLimitCount = "<<speedLimitCount<<endl;
		cout<<"genericCount = "<<genericCount<<endl;
	}






/*
  vector<Mat> rgb_planes;
  split(imgWb, rgb_planes);
  reds = reds/255;
  rgb_planes[0] = rgb_planes[0].mul(reds);
  rgb_planes[1] = rgb_planes[1].mul(reds);
  rgb_planes[2] = rgb_planes[2].mul(reds);
  Mat imgP;
  merge(rgb_planes, imgP);
	if(verbosity>=VERBOSITY_SHOW_IMAGES)
  	displayImage(imgP,"preprocessed image",1000);

  Mat imgGray;
  Mat imgGrayNeq;
  cvtColor(imgWb, imgGrayNeq, CV_BGR2GRAY);
  equalizeHist(imgGrayNeq, imgGray);

	if(verbosity>=VERBOSITY_SHOW_IMAGES)
  	displayImage(imgGray,"grayscale",1000);
*/
/*
  Mat edgeImg;
  getEdges(imgP, edgeImg);
  displayImage(edgeImg,"edges with preproc",1000);
  getEdges(imgGray, edgeImg);
  displayImage(edgeImg,"edges with grayscale",1000);
  edgeImg = edgeImg.mul(reds);
  displayImage(edgeImg,"masked edges",1000);


	Mat imgP_gray;
	cvtColor(imgP, imgP_gray, CV_BGR2GRAY);
  std::vector<Vec3f> circles;
  Mat imgCirc;
  imgWb.copyTo(imgCirc);
  getCircles(imgP_gray, circles);
  drawCirclesOnImg(circles, imgCirc, 15);
  displayImage(imgCirc,"circles with preproc",1000);
*/
/*
  circles.clear();
  imgWb.copyTo(imgCirc);
  getCircles(imgGray, circles);
  drawCirclesOnImg(circles, imgCirc, 15);
  displayImage(imgCirc,"circles with gray",1000);*/



	std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
	unsigned long identifyDuration =  std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() - wbVisualizationDuration - redsVisualizationDuration;
	if(verbosity>=VERBOSITY_TEXT_ONLY)
	{
		cout<<"identify required "<<identifyDuration<<" ms"<<endl;
		if(verbosity>=VERBOSITY_SHOW_MORE_IMAGES)
			cout<<"please not that with verbose image visualization some durations are invalid as the visualization time is included"<<endl;
	}

  return detectedSigns;
}

/**
 * [StreetSignsIdentifier::findReds description]
 * @param  img [description]
 * @return     [description]
 */
cv::Mat StreetSignsIdentifier::findReds(const cv::Mat& img)
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  Mat imgR1 = findRedsV1(img);
  Mat imgR2 = findRedsV2(img);

  resize(imgR1,imgR1,img.size(),0,0,INTER_LINEAR);
  resize(imgR2,imgR2,img.size(),0,0,INTER_LINEAR);

	std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
	unsigned long redsDetectionDuration =  std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	if(verbosity>=VERBOSITY_TEXT_ONLY)
		cout<<"findReds required "<<redsDetectionDuration<<" ms"<<endl;
	if(verbosity>=VERBOSITY_SHOW_MORE_IMAGES)
	{
		displayImage(imgR1,"findRedsV1",1000);
		displayImage(imgR2,"findRedsV2",1000);
	}
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
	if(img.cols<200)
		resize(img,imgResized,Size(50,50));
	else if(img.cols<50)
		img.copyTo(imgResized);
	else
  	resize(img,imgResized,Size(),0.25,0.25);

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  Mat blurredImg;
  GaussianBlur(imgResized,blurredImg, Size(5,5), 1);


  unsigned long blurTime =  std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();
  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"findRedsV1: blur took: "<<blurTime<<" us"<<endl;
  //find the red pixel by analyzing the hue


  Mat imgHls;
  cvtColor(blurredImg,imgHls,COLOR_BGR2HLS);
  vector<cv::Mat> hls_planes;
  split( imgHls, hls_planes );

  unsigned long hsvTime =  std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() - blurTime;
  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"findRedsV1: hsv conversion took: "<<hsvTime<<"ms"<<endl;

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



  unsigned long redDistTime =  std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() - hsvTime;
  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"findRedsV1: red Dist computation took: "<<redDistTime<<" ms"<<endl;

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
    cout<<"findRedsV1 took "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<" ms"<<endl;
  //if you want to see the intermediate images
  if(verbosity>=VERBOSITY_SHOW_MORE_IMAGES)
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
	if(img.cols<200)
		resize(img,imgResized,Size(50,50));
	else if(img.cols<50)
		img.copyTo(imgResized);
	else
  	resize(img,imgResized,Size(),0.25,0.25);

  Mat blurredImg;
  GaussianBlur(imgResized,blurredImg, Size(5,5), 1);

  std::chrono::steady_clock::time_point blurredTime = std::chrono::steady_clock::now();
  //equalize the lightness channel
  Mat imgLab;
  cvtColor(blurredImg,imgLab,COLOR_BGR2Lab);

  std::chrono::steady_clock::time_point toLabTime = std::chrono::steady_clock::now();

  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"findRedsV2: lab conversion took: "<<std::chrono::duration_cast<std::chrono::milliseconds>(toLabTime - blurredTime).count()<<" ms"<<endl;


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
    cout<<"findRedsV2: dists computation took: "<<std::chrono::duration_cast<std::chrono::milliseconds>(afterDistsComputation - beforeDistsComputation).count()<<" ms"<<endl;


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
    cout<<"findRedsV2 took: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<" ms"<<endl;


  if(verbosity>=VERBOSITY_SHOW_MORE_IMAGES)
  {
    displayImage(res,"res",1000);
    displayImage(imgLab,"imgLab",1000);
  }


  return resThresh;
}

/**
 * gets the edges int he image using canny
 * @param inputImg [description]
 * @param edgeImg  [description]
 */
void StreetSignsIdentifier::getEdges(const cv::Mat& inputImg, cv::Mat& edgeImg)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  	Canny(inputImg,edgeImg,100,400);
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    unsigned long edgeDetectionTimeMillis =  std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    if(verbosity>=VERBOSITY_TEXT_ONLY)
      cout<<"edge detection required "<<edgeDetectionTimeMillis<<" ms"<<endl;
}

/**
 * gets circles int the image using hough
 * @param inputImg [description]
 * @param circles  [description]
 */
void StreetSignsIdentifier::getCircles(const cv::Mat& inputImg, std::vector<cv::Vec3f>& circles)
{
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  HoughCircles(inputImg, circles, HOUGH_GRADIENT,4, inputImg.cols/30, 200, 25, inputImg.cols/50, inputImg.cols*0.75 );
  std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
  unsigned long circlesDetectionTimeMillis =  std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"circles detection required "<<circlesDetectionTimeMillis<<" ms"<<endl;
}

/**
 * Gets the lines int he image using hough
 * @param inputEdgeImg [description]
 * @param lines        [description]
 */
void StreetSignsIdentifier::getLines(const cv::Mat& inputEdgeImg, std::vector<cv::Vec2f>& lines)
{
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	double rho = inputEdgeImg.cols*0.05;
	if(rho<1)
		rho =1;
  HoughLines(inputEdgeImg, lines, rho,CV_PI/180*(15),0.1*inputEdgeImg.cols*1.41*(inputEdgeImg.cols*0.01+1));
  std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
  unsigned long circlesDetectionTimeMicro =  std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
  if(verbosity>=VERBOSITY_TEXT_ONLY)
  {
		cout<<"lines detection required "<<circlesDetectionTimeMicro<<" us"<<endl;
		cout<<"detected "<<lines.size()<<" lines"<<endl;
	}
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
 * draws the provided lines on the provided image
 * @param lines          [description]
 * @param linesImage     [description]
 * @param linesToDrawNum [description]
 */
void StreetSignsIdentifier::drawLinesOnImg(std::vector<cv::Vec2f>& lines, cv::Mat& linesImage, int linesToDrawNum)
{
 //If the image is grayscale I convert it to rgb
 if(linesImage.channels()==1)
 {
	 cv::Mat imgRgb;
	 cv::cvtColor(linesImage, imgRgb, cv::COLOR_GRAY2BGR);
	 linesImage = imgRgb;
 }
 else if(linesImage.channels()!=3)
 {
	 THROW_NICE(invalid_argument,"unsupported image (has "+std::to_string(linesImage.channels())+" channels)");
 }

 for(unsigned int i=0;i<min((unsigned int)linesToDrawNum,(unsigned int)lines.size());i++)
	{

		double rho = lines.at(i)[0];
		double theta = lines.at(i)[1];

		double cost = cos(theta/*+CV_PI/2*/);
		double sint = sin(theta/*+CV_PI/2*/);
		double x0 = cost*rho;
		double y0 = sint*rho;
		cv::Point2f p1(x0 + 2000*(-sint), y0 + 2000*cost);
		cv::Point2f p2(x0 - 2000*(-sint), y0 - 2000*cost);
		//cout<<" p1 "<<p1<<endl;
		//cout<<" p2 "<<p2<<endl;
		cv::line(	linesImage,
					p1,
					p2,
					cv::Scalar(0,255,255),
					2,
					CV_AA);
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

	float minSize = inputImage.cols/25;
	float maxSize = inputImage.cols/2;
  classifier.detectMultiScale( convertedImage, detections,1.05,15,0,Size2f(minSize,minSize),Size2f(maxSize,maxSize));
}

/**
 * Detects triangulare warning signs
 * @param inputImage    [description]
 * @param detectedSigns [description]
 */
void StreetSignsIdentifier::detectWarningSigns(cv::Mat& inputImage, std::vector<StreetSign_Warning>& detectedSigns)
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	vector<Rect> detections;
	detectWithCascade(inputImage,warningSignsClassifier, detections);

	//build the descritpions
	for(Rect r : detections)
		detectedSigns.push_back(StreetSign_Warning(Point2f(r.x+r.width/2,r.y+r.height/2),r.size()));

  std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
  unsigned long warningsDetectionDuration =  std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"warnings detection required "<<warningsDetectionDuration<<" ms"<<endl;
}

/**
 * Detects round red signs
 * @param inputImage    [description]
 * @param detectedSigns [description]
 */
void StreetSignsIdentifier::detectRoundRedSigns(cv::Mat& inputImage, std::vector<StreetSign>& detectedSigns)
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	vector<Rect> detections;
	detectWithCascade(inputImage,roundSignsClassifier, detections);

	//build the descritpions
	for(Rect r : detections)
		detectedSigns.push_back(StreetSign(Point2f(r.x+r.width/2,r.y+r.height/2),r.size(),StreetSign::SignType::GENERIC));

  std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
  unsigned long detectionDuration =  std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  if(verbosity>=VERBOSITY_TEXT_ONLY)
    cout<<"no parking signs detection required "<<detectionDuration<<" ms"<<endl;
}

/**
 * Searches for a speed limit in the provided sign. If it isn't found the method returns -1, if it is found it returns the limit
 * @param  inputImage [description]
 * @return            [description]
 */
int StreetSignsIdentifier::searchSpeedLimit(const cv::Mat& inputImage)
{
	if(verbosity>=VERBOSITY_TEXT_ONLY)
		cout<<"searchSpeedLimit()..."<<endl;
	Mat convertedImage;
	if(inputImage.type()==CV_8UC3)
		cvtColor(inputImage, convertedImage, COLOR_BGR2GRAY);
	else if(inputImage.type()==CV_8U)
		convertedImage = inputImage;
	else
		THROW_NICE(invalid_argument,"provided image has unsupported format, only CV_8U and CV_8UC3 are supported");

	convertedImage = 255-convertedImage;
	Mat reds = findReds(inputImage);
	reds = reds/255;
	reds = 1-reds;
	if(verbosity>=VERBOSITY_SHOW_MORE_IMAGES)
		displayImage(reds,"reds",500);

//	vector< vector <Point> > redsContours; // Vector for storing contour
//	findContours( reds, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );



	Mat threshImg;
	threshold(convertedImage, threshImg, 200, 255, THRESH_BINARY | THRESH_OTSU);
	threshImg = threshImg.mul(reds);

	//erode(threshImg,threshImg,Mat());
	if(verbosity>=VERBOSITY_SHOW_IMAGES)
		displayImage(threshImg,"threshImg",500);

	vector< vector <Point> > contours; // Vector for storing contour
	vector< Vec4i > hierarchy;

	//Create input sample by contour finding and cropping
	findContours( threshImg, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
	if(verbosity>=VERBOSITY_TEXT_ONLY)
		cout<<"found "<<contours.size()<<" contours"<<endl;
	//Mat dst(src.rows,src.cols,CV_8UC3,Scalar::all(0));

	vector<int> maybeDigits;
	vector<Rect> maybeDigitsBB;


	for(unsigned  int i = 0; i< contours.size(); i=hierarchy[i][0] ) // iterate through each contour for first hierarchy level .
	{
	    Rect r= boundingRect(contours[i]);
			if(r.width>inputImage.cols/2 || r.width<inputImage.cols/20 || r.height<inputImage.rows/5 || (r.width/(float)r.height)>0.9 || (r.width/(float)r.height)<0.15)
				continue;
			maybeDigits.push_back(i);
			maybeDigitsBB.push_back(r);
	}

	if(verbosity>=VERBOSITY_TEXT_ONLY)
		cout<<"kept "<<maybeDigits.size()<<" contours"<<endl;

	if(verbosity>=VERBOSITY_SHOW_MORE_IMAGES)
	{
		for(unsigned int i=0;i<maybeDigits.size();i++)
		{
			Rect r = maybeDigitsBB.at(i);
			cout<<"contour = "<<r<<endl;
			Mat tmp;
			inputImage.copyTo(tmp);
			rectangle(tmp,r,Scalar(255,0,0),3);
			displayImage(tmp,"tmp",500);
		}
	}

	if(maybeDigits.size()>3 || maybeDigits.size()<2)
		return -1;

	int maxHeight = 0;
	int minHeight = 100000000;
	for(unsigned int i= 0;i<maybeDigits.size();i++)
	{
		if(maybeDigitsBB.at(i).height>maxHeight)
			maxHeight = maybeDigitsBB.at(i).height;
		if(maybeDigitsBB.at(i).height<minHeight)
			minHeight = maybeDigitsBB.at(i).height;
	}
	//cout<<"maxh="<<maxHeight<<"  minh="<<minHeight<<endl;
	if((maxHeight-minHeight)/(float)maxHeight>0.1)
		return -1;


	//order left to right
	for(unsigned int i= 0;i<maybeDigits.size();i++)
	{
		for(unsigned int j= i;j<maybeDigits.size();j++)
		{
			if(maybeDigitsBB.at(j).x<maybeDigitsBB.at(i).x)
			{
				int t = maybeDigits.at(i);
				maybeDigits.at(i)=maybeDigits.at(j);
				maybeDigits.at(j)=t;

				auto tr = maybeDigitsBB.at(i);
				maybeDigitsBB.at(i)=maybeDigitsBB.at(j);
				maybeDigitsBB.at(j)=tr;
			}
		}
	}

	int number=0;
	for(unsigned int i=0;i<maybeDigits.size();i++)
	{
		Rect r = maybeDigitsBB.at(i);
		//cout<<"contour = "<<r<<endl;
		if(verbosity>=VERBOSITY_SHOW_MORE_IMAGES)
		{
			Mat tmp;
			inputImage.copyTo(tmp);
			rectangle(tmp,r,Scalar(255,0,0),3);
			displayImage(tmp,"tmp",500);
		}

		Mat ROI = threshImg(r);
    Mat tmp1, tmp2;
    resize(ROI,tmp1, Size(10,10), 0,0,INTER_LINEAR );
    tmp1.convertTo(tmp2,CV_32FC1);
		Mat results;
    float p=digitsKnnPtr->findNearest(tmp2.reshape(1,1), 1,results);
		if((maybeDigits.size()-1-i)==2)
			p = 1;//brutto ma infallibile

		number += pow(10,maybeDigits.size()-1 - i)*p;
	}
	if(verbosity>=VERBOSITY_TEXT_ONLY)
		cout<<"number = "<<number<<endl;
	if(number-((number/10)*10)!=0)
		number=-1;
	return number;
}

/**
 * Classifies the provided round red signs in a more specific sign
 * @param inputImage      the whole image
 * @param inputOutputSign the sign to be classified
 */
void StreetSignsIdentifier::classifyRoundRedSign(cv::Mat& inputImage, std::shared_ptr<StreetSign>& inputOutputSign)
{
	//cout<<"--------------classifyRoundRedSign----------------"<<endl;
	Mat edgeImg;
	Mat signReds = findReds(inputImage(inputOutputSign->getBoundingBox()));
	getEdges(signReds, edgeImg);
	vector<Vec2f> lines;
	getLines(edgeImg, lines);


	int diagLinesCount = 0;
	double maxRho = inputOutputSign->getBoundingBox().width*0.20;
	double thetaTolerance = 7.5;
	double minTheta = CV_PI/180.0*(135-thetaTolerance);
	double maxTheta = CV_PI/180.0*(135+thetaTolerance);
	int c = 0;
	for(Vec2f l : lines)
	{
		if(c++>3)
			break;//look just the first 3 lines
	//	cout<<"line has "<<l[0]<<";"<<l[1]<<endl;
	//	cout<<"limits are: "<<maxRho<<" and "<<minTheta<<";"<<maxTheta<<endl;
		if(abs(l[0])<maxRho && (minTheta<l[1] && l[1]<maxTheta))
		{
		//	cout<<"it's diagonal"<<endl;
			diagLinesCount++;
		}
		else
		{
		//	cout<<"it's not diagonal"<<endl;
		}

	}
	if(diagLinesCount>0)
	{
		inputOutputSign = std::make_shared<StreetSign_NoParking>(inputOutputSign->getCenterPosition(),inputOutputSign->getBoundingBoxSize());
	//	cout<<"it's no parking"<<endl;
	}
	else
	{
		float inc = inputOutputSign->getBoundingBox().width*0.0;
		float x1 = inputOutputSign->getBoundingBox().tl().x -inc;
		if(x1<0)
			x1=0;
		float y1 = inputOutputSign->getBoundingBox().tl().y -inc;
		if(y1<0)
			y1=0;
		float x2 = inputOutputSign->getBoundingBox().br().x +inc;
		if(x2>=inputImage.cols)
			x2=inputImage.cols-1;
		float y2 = inputOutputSign->getBoundingBox().br().y +inc;
		if(y2>=inputImage.rows)
			y2=inputImage.rows-1;
		Rect enlargedBB(Point(x1,y1),Point(x2,y2));
		int speedLimit = searchSpeedLimit(inputImage(enlargedBB));
		if(speedLimit>0)
		{
		//	cout<<"it's speed limit"<<endl;
			inputOutputSign = std::make_shared<StreetSign_Speed>(inputOutputSign->getCenterPosition(),inputOutputSign->getBoundingBoxSize(),speedLimit);
		}
	}

	if(verbosity>=VERBOSITY_SHOW_MORE_IMAGES)
	{
		displayImage(edgeImg,"sign edges",500);
		Mat imgWithLines;
		inputImage(inputOutputSign->getBoundingBox()).copyTo(imgWithLines);
		drawLinesOnImg(lines, imgWithLines, 4);
		displayImage(imgWithLines,"sign lines",500);
	}
}

/**
 * filters out the detected signs that have less than 5% of thei pixel which intersect the mask
 * @param detectedSigns [description]
 * @param binaryMask    [description]
 */
void StreetSignsIdentifier::filterDetectionsByBinaryMask(std::vector<std::shared_ptr<StreetSign>>& detectedSigns, cv::Mat& binaryMask)
{
	for(unsigned int i=0;i<detectedSigns.size();i++)
	{
		shared_ptr<StreetSign> ssp = detectedSigns.at(i);
		if(countNonZero(binaryMask(ssp->getBoundingBox()))<=ssp->getBoundingBox().area()*0.05)
		{
			detectedSigns.erase(detectedSigns.begin()+i);
			i--;
		}
	}
}

/**
 * filters out signs which are inside other signs
 * @param detectedSigns [description]
 */
void StreetSignsIdentifier::filterSignsContainedInBiggerOnes(std::vector<std::shared_ptr<StreetSign>>& detectedSigns)
{
	for(unsigned int i=0;i<detectedSigns.size();i++)
	{
		shared_ptr<StreetSign> innerSign = detectedSigns.at(i);
		for(unsigned int o=0;o<detectedSigns.size();o++)
		{
			shared_ptr<StreetSign> outerSign = detectedSigns.at(o);
			if(o!=i)
			{
				if(outerSign->getBoundingBox().width>innerSign->getBoundingBox().width &&
						outerSign->getBoundingBox().contains(innerSign->getCenterPosition()))
				{
						detectedSigns.erase(detectedSigns.begin()+i);
						i--;
				}
			}
		}
	}
}

/**
 * Build the kNearesNeighbor classifier for classifiing the digits in the speed limti  signs
 */
void StreetSignsIdentifier::buildKnn()
{
	string path = "./templates/digitsSmall.png";
	Mat image = cv::imread(path,CV_LOAD_IMAGE_GRAYSCALE);
	if(! image.data )
		THROW_NICE(runtime_error,"digits.png (the image from the opencv samples) not found at "+path);

	Mat thrImg;
	threshold(image,thrImg,200,255,THRESH_BINARY_INV); //Threshold to find contour

	Mat sample;
	Mat response_array;
/*
//opencv's digits.png
//20px by 20px each digit, the image is 2000x1000 (width x height), 5 rows for each digit
//so 100 digit per line, 50 lines
	for(int x = 0; x<100;x++)
	{
		for(int y=0;y<50;y++)
		{
			Rect r(Point(x*20,y*20),Size(20,20));
			Mat ROI = thrImg(r); //Crop the image
	    Mat tmp1, tmp2;
	    resize(ROI,tmp1, Size(10,10), 0,0,INTER_LINEAR ); //resize to 10X10
	    tmp1.convertTo(tmp2,CV_32FC1); //convert to float
	    sample.push_back(tmp2.reshape(1,1)); // Store  sample data
			response_array.push_back((int)(y/5));
		}
	}
*/
	for(int x = 0; x<10;x++)
	{
			Rect r(Point(x*16,0),Size(16,24));
			//cout<<"r="<<r<<endl;
			Mat ROI = thrImg(r); //Crop the image
	    Mat tmp1, tmp2;
	    resize(ROI,tmp1, Size(10,10), 0,0,INTER_LINEAR ); //resize to 10X10
	    tmp1.convertTo(tmp2,CV_32FC1); //convert to float
	    sample.push_back(tmp2.reshape(1,1)); // Store  sample data
			response_array.push_back(x);
	}

	Mat response,tmp;
	tmp=response_array.reshape(1,1); //make continuous
	tmp.convertTo(response,CV_32FC1); // Convert  to float

	digitsKnnPtr = cv::ml::KNearest::create();
	digitsKnnPtr->train(sample,cv::ml::SampleTypes::ROW_SAMPLE,response);
}
