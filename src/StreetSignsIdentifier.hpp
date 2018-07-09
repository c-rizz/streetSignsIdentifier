#ifndef STREET_SIGNS_STREET_SIGNS_IDENTIFIER
#define STREET_SIGNS_STREET_SIGNS_IDENTIFIER

#include <vector>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <memory>
#include "StreetSign.hpp"
#include <opencv2/objdetect.hpp>
#include "StreetSign_Warning.hpp"
#include "StreetSign_Speed.hpp"
#include "StreetSign_NoParking.hpp"

class StreetSignsIdentifier
{
private:
  int verbosity = 0;
  cv::CascadeClassifier warningSignsClassifier;
  cv::CascadeClassifier speedLimitSignsClassifier;
  cv::CascadeClassifier noParkingSignsClassifier;

  cv::Mat findReds(const cv::Mat& img);
  cv::Mat findRedsV1(const cv::Mat& img);
  cv::Mat findRedsV2(const cv::Mat& img);

  void getEdges(const cv::Mat& inputImg, cv::Mat& edgeImg);

  void getCircles(const cv::Mat& inputImg, std::vector<cv::Vec3f>& circles);

  void drawCirclesOnImg(std::vector<cv::Vec3f>& circles, cv::Mat& circlesImage, int circlesToDrawNum);

  void detectWithCascade(cv::Mat& inputImage, cv::CascadeClassifier& classifier, std::vector<cv::Rect>& detections);

  void detectWarningSigns(cv::Mat& inputImage, std::vector<StreetSign_Warning>& detectedSigns);
  void detectSpeedLimitSigns(cv::Mat& inputImage, std::vector<StreetSign_Speed>& detectedSigns);
  void detectNoParkingSigns(cv::Mat& inputImage, std::vector<StreetSign_NoParking>& detectedSigns);

  void detectAllSigns(cv::Mat& inputImage, std::vector<std::shared_ptr<StreetSign>>& detectedSigns);

  void filterDetectionsByBinaryMask(std::vector<std::shared_ptr<StreetSign>>& detectedSigns, cv::Mat& binaryMask);
  void filterSignsContainedInBiggerOnes(std::vector<std::shared_ptr<StreetSign>>& detectedSigns);

public:
  StreetSignsIdentifier(std::string warningSignsClassifierPath,std::string speedLimitSignsClassifierPath,std::string noParkingSignsClassifierPath);

  static const int VERBOSITY_NORMAL;
  static const int VERBOSITY_TEXT_ONLY;
  static const int VERBOSITY_SHOW_IMAGES;
  static const int VERBOSITY_SHOW_MORE_IMAGES;
  void setVerbosity(int status);

  std::vector<std::shared_ptr<StreetSign>> identify(cv::Mat img);
};

#endif
