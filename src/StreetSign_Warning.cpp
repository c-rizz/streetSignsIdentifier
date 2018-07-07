#include "StreetSign_Warning.hpp"

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
