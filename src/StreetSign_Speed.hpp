#ifndef STREET_SIGNS_STREET_SIGN_SPEED_HPP
#define STREET_SIGNS_STREET_SIGN_SPEED_HPP

#include <opencv2/core/core.hpp>
#include "StreetSign.hpp"

class StreetSign_Speed : public StreetSign
{
private:
  int speedLimit;
public:
  StreetSign_Speed(const cv::Point2f& centerPosition, const cv::Size2f& boundingBoxSize, int speedLimit);
  virtual void drawOnImage(cv::Mat& inputOutputImage);
  int getLimit();
};

#endif
