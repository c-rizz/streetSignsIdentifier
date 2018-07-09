#include "StreetSign.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

/**
 * Constructs the StreetSign
 * @param centerPosition The pixel position of the center of the sign
 * @param height         The max vertical pixel size of the sign
 * @param type           the type of sign, see StreetSign::SignType
 */
StreetSign::StreetSign(const cv::Point2f& centerPosition, const cv::Size2f& boundingBoxSize, SignType type) :
  centerPosition(centerPosition),
  boundingBoxSize(boundingBoxSize),
  type(type)
{
}

/**
 * Get the pixel position of the center of the sign in the image
 * @return [description]
 */
cv::Point2f StreetSign::getCenterPosition()
{
  return centerPosition;
}

/**
 * Gets the size of the bounding box for the sign, centered on \ref centerPosition
 * @return the bounding box size
 */
cv::Size2f StreetSign::getBoundingBoxSize()
{
  return boundingBoxSize;
}

/**
 * Gets the type of this sign
 * @return see \ref StreetSign::SignType
 */
StreetSign::SignType StreetSign::getType()
{
  return type;
}

/**
 * Draws a bounding symbol around the sign
 * @param inputOutputImage image to draw upon
 */
void StreetSign::drawOnImage(cv::Mat& inputOutputImage)
{
  rectangle(inputOutputImage,
      cv::Point2f(centerPosition.x-boundingBoxSize.width/2,centerPosition.y-boundingBoxSize.height/2),
      cv::Point2f(centerPosition.x+boundingBoxSize.width/2,centerPosition.y+boundingBoxSize.height/2),
      cv::Scalar(128,128,128), inputOutputImage.rows/400+1);
}


/**
 * Returns the traffic sign bounding box
 * @return the bounding box
 */
Rect StreetSign::getBoundingBox()
{
  return Rect((int)(centerPosition.x-boundingBoxSize.width/2),
              (int)(centerPosition.y-boundingBoxSize.height/2),
              (int)(boundingBoxSize.width),
              (int)(boundingBoxSize.height));
}
