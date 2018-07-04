#ifndef STREET_SIGNS_STREET_SIGN_HPP
#define STREET_SIGNS_STREET_SIGN_HPP

#include <opencv2/core/core.hpp>

class StreetSign
{
public:
  /**
   * Possible types of street signs
   */
  enum class SignType{SPEED,WARNING,NO_PARKING};
private:
  /**
   * Center of the sign in the image
   */
  const cv::Point2f centerPosition;

  /**
   * Size of the bounding box of the street sign, centered on \ref centerPosition
   */
  const cv::Size2f boundingBoxSize;

  /**
   * type of street sign
   */
  const SignType type;
protected:
  StreetSign(cv::Point2f& centerPosition, cv::Size2f& boundingBoxSize, SignType type);

public:
  cv::Point2f getCenterPosition();
  cv::Size2f getBoundingBoxSize();
  SignType getType();
};

#endif
