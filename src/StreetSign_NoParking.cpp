#include "StreetSign_NoParking.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

/**
 * Builds a no parking sign description
 * @param centerPosition  the position of the center of the sign
 * @param boundingBoxSize the size of the bounding box
 */
StreetSign_NoParking::StreetSign_NoParking(const cv::Point2f& centerPosition, const cv::Size2f& boundingBoxSize) :
  StreetSign(centerPosition, boundingBoxSize, StreetSign::SignType::NO_PARKING)
{
  //nothing to do for now
}


/**
 * Draws a bounding symbol around the sign
 * @param inputOutputImage image to draw upon
 */
void StreetSign_NoParking::drawOnImage(cv::Mat& inputOutputImage)
{
  circle(inputOutputImage,
      getCenterPosition(),
      getBoundingBoxSize().width/2,
      cv::Scalar(0,0,255), inputOutputImage.rows/200+1);
}
