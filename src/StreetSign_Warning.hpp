#ifndef STREET_SIGNS_STREET_SIGN_WARNING_HPP
#define STREET_SIGNS_STREET_SIGN_WARNING_HPP

#include <opencv2/core/core.hpp>
#include "StreetSign.hpp"

class StreetSign_Warning : public StreetSign
{
public:
  StreetSign_Warning(const cv::Point2f& centerPosition, const cv::Size2f& boundingBoxSize);
};

#endif
