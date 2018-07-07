#ifndef STREET_SIGNS_UTILS_HPP
#define STREET_SIGNS_UTILS_HPP

#include <string>
#include <opencv2/core/core.hpp>

void displayImage(cv::Mat& image, std::string winName, int winWidth, int winHeight=-1);

#define THROW_NICE(exception, msg) throw exception(__FILE__+std::string(":")+to_string(__LINE__)+" ["+__FUNCTION__+"]: "+(msg));
#endif
