#ifndef STREET_SIGNS_STREET_SIGN_NO_PARKING_HPP
#define STREET_SIGNS_STREET_SIGN_NO_PARKING_HPP

#include <opencv2/core/core.hpp>
#include "StreetSign.hpp"

class StreetSign_NoParking : public StreetSign
{
public:
  StreetSign_NoParking(const cv::Point2f& centerPosition, const cv::Size2f& boundingBoxSize);
  virtual void drawOnImage(cv::Mat& inputOutputImage) override;
};

#endif
