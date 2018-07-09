#ifndef STREET_SIGNS_STREET_SIGNS_IDENTIFIER
#define STREET_SIGNS_STREET_SIGNS_IDENTIFIER

#include <vector>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
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
  cv::CascadeClassifier roundSignsClassifier;
  cv::Ptr<cv::ml::KNearest> digitsKnnPtr;


  cv::Mat findReds(const cv::Mat& img);
  cv::Mat findRedsV1(const cv::Mat& img);
  cv::Mat findRedsV2(const cv::Mat& img);

  void getEdges(const cv::Mat& inputImg, cv::Mat& edgeImg);

  void getCircles(const cv::Mat& inputImg, std::vector<cv::Vec3f>& circles);
  void getLines(const cv::Mat& inputEdgeImg, std::vector<cv::Vec2f>& lines);

  void drawCirclesOnImg(std::vector<cv::Vec3f>& circles, cv::Mat& circlesImage, int circlesToDrawNum);
  void drawLinesOnImg(std::vector<cv::Vec2f>& lines, cv::Mat& linesImage, int linesToDrawNum);

  void detectWithCascade(cv::Mat& inputImage, cv::CascadeClassifier& classifier, std::vector<cv::Rect>& detections);

  void detectWarningSigns(cv::Mat& inputImage, std::vector<StreetSign_Warning>& detectedSigns);
  void detectRoundRedSigns(cv::Mat& inputImage, std::vector<StreetSign>& detectedSigns);
  void classifyRoundRedSign(cv::Mat& inputImage, std::shared_ptr<StreetSign>& detectedSign);
  int searchSpeedLimit(const cv::Mat& inputImage);


  void filterDetectionsByBinaryMask(std::vector<std::shared_ptr<StreetSign>>& detectedSigns, cv::Mat& binaryMask);
  void filterSignsContainedInBiggerOnes(std::vector<std::shared_ptr<StreetSign>>& detectedSigns);
  void buildKnn();

public:
  StreetSignsIdentifier(std::string warningSignsClassifierPath,std::string speedLimitSignsClassifierPath,std::string roundSignsClassifierPath);

  static const int VERBOSITY_NORMAL;
  static const int VERBOSITY_TEXT_ONLY;
  static const int VERBOSITY_SHOW_IMAGES;
  static const int VERBOSITY_SHOW_MORE_IMAGES;
  void setVerbosity(int status);

  std::vector<std::shared_ptr<StreetSign>> identify(cv::Mat img);
};

#endif
