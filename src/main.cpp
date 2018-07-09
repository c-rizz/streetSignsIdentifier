
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>

#include "StreetSignsIdentifier.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
  if(argc<4)
  {
    cout<<"Invalid parameters."<<endl;
    cout<<"correct usage is:"<<endl;
    cout<<argv[0]<<" <inputImage> <warningClassifier> <roundSignsClassifier> [options]"<<endl;
    return -1;
  }

  int verbosity = StreetSignsIdentifier::VERBOSITY_NORMAL;
  for(int i=1;i<argc;i++)
  {
    if(strcmp(argv[i],"-v")==0)
      verbosity=std::max(StreetSignsIdentifier::VERBOSITY_TEXT_ONLY,verbosity);
    if(strcmp(argv[i],"-vv")==0)
      verbosity=std::max(StreetSignsIdentifier::VERBOSITY_SHOW_IMAGES,verbosity);
    if(strcmp(argv[i],"-vvv")==0)
      verbosity=std::max(StreetSignsIdentifier::VERBOSITY_SHOW_MORE_IMAGES,verbosity);
  }
  string path = argv[1];
  string warningClassifierPath = argv[2];
  string noParkingClassifierPath = argv[3];

  cv::Mat image = cv::imread(path);
	if(! image.data )
	{
		cout <<"image not found at "<<path << endl ;
		return -1;
	}

  displayImage(image,"image",1000,1000*image.rows/image.cols);

  StreetSignsIdentifier ssi(warningClassifierPath,"",noParkingClassifierPath);
  ssi.setVerbosity(verbosity);

  std::vector<std::shared_ptr<StreetSign>> streetSigns = ssi.identify(image);

  for(shared_ptr<StreetSign> s : streetSigns)
  {
    s->drawOnImage(image);
    if(s->getType()==StreetSign::SignType::WARNING)
    {
      cout<<"Detected a warning sign"<<endl;
    }
    else if(s->getType()==StreetSign::SignType::NO_PARKING)
    {
      cout<<"Detected a no parking sign"<<endl;
    }
    else if(s->getType()==StreetSign::SignType::SPEED)
    {
      std::shared_ptr<StreetSign_Speed> sssp = std::dynamic_pointer_cast<StreetSign_Speed> (s);
      cout<<"Detected a speed limit sign, limit at "<<sssp->getLimit()<<endl;
    }
  }

  displayImage(image,"image",1000,1000*image.rows/image.cols);

}
