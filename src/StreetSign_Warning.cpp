#include "StreetSign_Warning.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

/**
 * Builds a warning sign description
 * @param centerPosition  the position of the center of the sign
 * @param boundingBoxSize the size of the bounding box
 */
StreetSign_Warning::StreetSign_Warning(const cv::Point2f& centerPosition, const cv::Size2f& boundingBoxSize) :
  StreetSign(centerPosition, boundingBoxSize, StreetSign::SignType::WARNING)
{
  //nothing to do for now
}


/**
 * Draws a bounding symbol around the sign
 * @param inputOutputImage image to draw upon
 */
void StreetSign_Warning::drawOnImage(cv::Mat& inputOutputImage)
{
  rectangle(inputOutputImage,
      cv::Point2f(getCenterPosition().x-getBoundingBoxSize().width/2,getCenterPosition().y-getBoundingBoxSize().height/2),
      cv::Point2f(getCenterPosition().x+getBoundingBoxSize().width/2,getCenterPosition().y+getBoundingBoxSize().height/2),
      cv::Scalar(0,0,255), inputOutputImage.rows/200+1);
}
