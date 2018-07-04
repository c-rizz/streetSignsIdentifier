#include "StreetSign.hpp"

/**
 * Constructs the StreetSign
 * @param centerPosition The pixel position of the center of the sign
 * @param height         The max vertical pixel size of the sign
 * @param type           the type of sign, see StreetSign::SignType
 */
StreetSign::StreetSign(cv::Point2f& centerPosition, cv::Size2f& boundingBoxSize, SignType type) :
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
