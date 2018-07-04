#ifndef STREET_SIGNS_STREET_SIGNS_IDENTIFIER
#define STREET_SIGNS_STREET_SIGNS_IDENTIFIER

#include <vector>
#include <chrono>
#include <opencv2/core/core.hpp>
#include "StreetSign.hpp"

class StreetSignsIdentifier
{
private:
  int verbosity = 0;
  unsigned long preprocessForRedDurationMicro = 0;
private:
  void preprocessForRedSigns(cv::Mat& img);

  void getEdges(const cv::Mat& inputImg, cv::Mat& edgeImg);

  void getCircles(const cv::Mat& inputImg, std::vector<cv::Vec3f>& circles);

  void drawCirclesOnImg(std::vector<cv::Vec3f>& circles, cv::Mat& circlesImage, int circlesToDrawNum);
public:
  StreetSignsIdentifier();

  static const int VERBOSITY_NORMAL;
  static const int VERBOSITY_TEXT_ONLY;
  static const int VERBOSITY_SHOW_IMAGES;
  void setVerbosity(int status);

  std::vector<StreetSign> identify(cv::Mat img);
};

#endif
