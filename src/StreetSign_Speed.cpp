#include "StreetSign_Speed.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

/**
 * Builds a no parking sign description
 * @param centerPosition  the position of the center of the sign
 * @param boundingBoxSize the size of the bounding box
 */
StreetSign_Speed::StreetSign_Speed(const cv::Point2f& centerPosition, const cv::Size2f& boundingBoxSize, int speedLimit) :
  StreetSign(centerPosition, boundingBoxSize, StreetSign::SignType::SPEED),speedLimit(speedLimit)
{
  //nothing to do for now
}


/**
 * Draws a bounding symbol around the sign
 * @param inputOutputImage image to draw upon
 */
void StreetSign_Speed::drawOnImage(cv::Mat& inputOutputImage)
{
  circle(inputOutputImage,
      getCenterPosition(),
      getBoundingBoxSize().width/2,
      cv::Scalar(0,255,255), inputOutputImage.rows/200+1);
}

/**
 * returns the speed limit for this sign
 * @return the limit
 */
int StreetSign_Speed::getLimit()
{
  return speedLimit;
}
