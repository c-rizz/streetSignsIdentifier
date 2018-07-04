
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>

#include "StreetSignsIdentifier.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
  if(argc<2)
  {
    cout<<"Invalid parameters."<<endl;
    cout<<"correct usage is:"<<endl;
    cout<<argv[0]<<" <inputImage> [options]"<<endl;
    return -1;
  }

  int verbosity = StreetSignsIdentifier::VERBOSITY_NORMAL;
  for(int i=1;i<argc;i++)
  {
    if(strcmp(argv[i],"-v")==0)
      verbosity=std::max(StreetSignsIdentifier::VERBOSITY_TEXT_ONLY,verbosity);
    if(strcmp(argv[i],"-vv")==0)
      verbosity=std::max(StreetSignsIdentifier::VERBOSITY_SHOW_IMAGES,verbosity);
  }
  string path = argv[1];

  cv::Mat image = cv::imread(path);
	if(! image.data )
	{
		cout <<"image not found at "<<path << endl ;
		return -1;
	}

  displayImage(image,"image",1000,1000*image.rows/image.cols);

  StreetSignsIdentifier ssi;
  ssi.setVerbosity(verbosity);

  ssi.identify(image);

}
