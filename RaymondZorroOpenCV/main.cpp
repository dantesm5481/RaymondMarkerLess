//此處為 使用到的功能都要進行 inculde
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>

// Another cpp
#include "ImagePath.hpp"
#include "PatternMatche.hpp"
#include "RLog.hpp"

using namespace cv;
using namespace std;


//void Rlog(string s){cout << s << endl;}

#define ClusterNum 10

int main(){
    
    cv::initModule_nonfree();//使用SIFT/SURF create之前，必須先initModule_<modulename>();
    
    //    cout << "< Creating detector, descriptor extractor and descriptor matcher ...";
    RLog("Creating detector, descriptor extractor and descriptor matcher ...");
    Ptr<FeatureDetector> detector = FeatureDetector::create( "SIFT" );
    
    Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create( "SIFT" );
    
    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( "BruteForce" );
    
    
    
    cout << ">" << endl;
    RLog(">");
    
    if( detector.empty() || descriptorExtractor.empty() )
    {
        
        RLog("Can not create detector or descriptor exstractor or descriptor matcher of given types");
        return -1;
    }
    RLog("<....Reading images....> ");
    Mat img1 = imread(getImgPath1());
    Mat img2 = imread(getImgPath2());
    
    cout<<endl<<">"<<endl;
    RLog(">");
    
    //detect keypoints;
    RLog("Extracting keypoints from images...");
    
    vector<KeyPoint> keypoints1,keypoints2;
    detector->detect( img1, keypoints1 );
    detector->detect( img2, keypoints2 );
    
    RLog(&"img1 : "[keypoints1.size()]);
    RLog(&"img2 : "[keypoints2.size()]);
    RLog(">");
    
    //compute descriptors for keypoints;
    RLog("Computing descriptors for keypoints from images...");
    
    Mat descriptors1,descriptors2;
    descriptorExtractor->compute( img1, keypoints1, descriptors1 );
    descriptorExtractor->compute( img2, keypoints2, descriptors2 );
    
    
    cout<<endl<<"< Descriptoers Size: "<<descriptors2.size()<<" >"<<endl;
    cout<<endl<<"descriptor's col: "<<descriptors2.cols<<endl
    <<"descriptor's row: "<<descriptors2.rows<<endl;
    cout << ">" << endl;
    
    
    
    //Draw And Match img1,img2 keypoints
    //匹配的過程是退特徵點的的descriptors進行match;
    Mat Match = DrawAndMatchKeypoints(img1,img2,keypoints1,keypoints2,descriptors1,descriptors2);
    
    //對img1提取特徵點，並且聚類，測試OpenCV：class BOWTrainer
    
    Mat BOW = BOWKeams(img1,keypoints1,descriptors1,*new Mat());
    
    
    imshow("Match", Match);
    imshow("BOWKeams", BOW);
    
    waitKey(0);
    
    
    
    
    return 0;
    
}








