////此處為 使用到的功能都要進行 inculde
//#include <iostream>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
//#include <opencv2/calib3d/calib3d.hpp>
//
//// Another cpp
//#include "ImagePath.hpp"
//#include "PatternMatche.hpp"
//#include "RLog.hpp"
//
//using namespace cv;
//using namespace std;
//
//
////void Rlog(string s){cout << s << endl;}
//
//#define ClusterNum 10
//
//int main(){
//    
//    cv::initModule_nonfree();//使用SIFT/SURF create之前，必須先initModule_<modulename>();
//    
//    //    cout << "< Creating detector, descriptor extractor and descriptor matcher ...";
//    RLog("Creating detector, descriptor extractor and descriptor matcher ...");
//    Ptr<FeatureDetector> detector = FeatureDetector::create( "SIFT" );
//    
//    Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create( "SIFT" );
//    
//    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( "BruteForce" );
//    
//    
//    
//    cout << ">" << endl;
//    RLog(">");
//    
//    if( detector.empty() || descriptorExtractor.empty() )
//    {
//        
//        RLog("Can not create detector or descriptor exstractor or descriptor matcher of given types");
//        return -1;
//    }
//    RLog("<....Reading images....> ");
//    Mat img1 = imread(getImgPath1());
//    Mat img2 = imread(getImgPath2());
//    
//    cout<<endl<<">"<<endl;
//    RLog(">");
//    
//    //detect keypoints;
//    RLog("Extracting keypoints from images...");
//    
//    vector<KeyPoint> keypoints1,keypoints2;
//    detector->detect( img1, keypoints1 );
//    detector->detect( img2, keypoints2 );
//    
//    RLog(&"img1 : "[keypoints1.size()]);
//    RLog(&"img2 : "[keypoints2.size()]);
//    RLog(">");
//    
//    //compute descriptors for keypoints;
//    RLog("Computing descriptors for keypoints from images...");
//    
//    Mat descriptors1,descriptors2;
//    descriptorExtractor->compute( img1, keypoints1, descriptors1 );
//    descriptorExtractor->compute( img2, keypoints2, descriptors2 );
//    
//    
//    cout<<endl<<"< Descriptoers Size: "<<descriptors2.size()<<" >"<<endl;
//    cout<<endl<<"descriptor's col: "<<descriptors2.cols<<endl
//    <<"descriptor's row: "<<descriptors2.rows<<endl;
//    cout << ">" << endl;
//    
//    
//    
//    //Draw And Match img1,img2 keypoints
//    //匹配的過程是退特徵點的的descriptors進行match;
//    Mat Match = DrawAndMatchKeypoints(img1,img2,keypoints1,keypoints2,descriptors1,descriptors2);
//    
//    //對img1提取特徵點，並且聚類，測試OpenCV：class BOWTrainer
//    
//    Mat BOW = BOWKeams(img1,keypoints1,descriptors1,*new Mat());
//    
//    
//    imshow("Match", Match);
//    imshow("BOWKeams", BOW);
//    
//    waitKey(0);
//    
//    
//    
//    
//    return 0;
//    
//}
//
//
//
//
//
//
//
//




#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <vector>
#include <iostream>
#include "ImagePath.hpp"

using namespace cv;
using namespace std;
Mat src,frameImg;
int width;
int height;
vector<Point> srcCorner(4);
vector<Point> dstCorner(4);

static bool createDetectorDescriptorMatcher( const string& detectorType, const string& descriptorType, const string& matcherType,
                                            Ptr<FeatureDetector>& featureDetector,
                                            Ptr<DescriptorExtractor>& descriptorExtractor,
                                            Ptr<DescriptorMatcher>& descriptorMatcher )
{
    cout << "< Creating feature detector, descriptor extractor and descriptor matcher ..." << endl;
    if (detectorType=="SIFT"||detectorType=="SURF")
        initModule_nonfree();
    featureDetector = FeatureDetector::create( detectorType );
    descriptorExtractor = DescriptorExtractor::create( descriptorType );
    descriptorMatcher = DescriptorMatcher::create( matcherType );
    cout << ">" << endl;
    
    bool isCreated = !( featureDetector.empty() || descriptorExtractor.empty() || descriptorMatcher.empty() );
    if( !isCreated )
        cout << "Can not create feature detector or descriptor extractor or descriptor matcher of given types." << endl << ">" << endl;
    
    return isCreated;
}


bool refineMatchesWithHomography(const std::vector<cv::KeyPoint>& queryKeypoints,
                                 const std::vector<cv::KeyPoint>& trainKeypoints,
                                 float reprojectionThreshold,
                                 std::vector<cv::DMatch>& matches,
                                 cv::Mat& homography  )
{
    const int minNumberMatchesAllowed = 4;
    if (matches.size() < minNumberMatchesAllowed)
        return false;
    // Prepare data for cv::findHomography
    std::vector<cv::Point2f> queryPoints(matches.size());
    std::vector<cv::Point2f> trainPoints(matches.size());
    for (size_t i = 0; i < matches.size(); i++)
    {
        
        queryPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
        trainPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
    }
    // Find homography matrix and get inliers mask
    std::vector<unsigned char> inliersMask(matches.size());
    homography = cv::findHomography(queryPoints,
                                    trainPoints,
                                    CV_FM_RANSAC,
                                    reprojectionThreshold,
                                    inliersMask);
    std::vector<cv::DMatch> inliers;
    for (size_t i=0; i<inliersMask.size(); i++)
    {
        if (inliersMask[i])
            inliers.push_back(matches[i]);
    }
    matches.swap(inliers);
    Mat homoShow;
    drawMatches(src,queryKeypoints,frameImg,trainKeypoints,matches,homoShow,Scalar::all(-1),CV_RGB(255,255,255),Mat(),2);
    imshow("homoShow",homoShow);
    return matches.size() > minNumberMatchesAllowed;
    
}


bool matchingDescriptor(const vector<KeyPoint>& queryKeyPoints,const vector<KeyPoint>& trainKeyPoints,
                        const Mat& queryDescriptors,const Mat& trainDescriptors,
                        Ptr<DescriptorMatcher>& descriptorMatcher,
                        bool enableRatioTest = true)
{
    vector<vector<DMatch>> m_knnMatches;
    vector<DMatch>m_Matches;
    
    if (enableRatioTest)
    {
        cout<<"KNN Matching"<<endl;
        const float minRatio = 1.f / 1.5f;
        descriptorMatcher->knnMatch(queryDescriptors,trainDescriptors,m_knnMatches,2);
        for (size_t i=0; i<m_knnMatches.size(); i++)
        {
            const cv::DMatch& bestMatch = m_knnMatches[i][0];
            const cv::DMatch& betterMatch = m_knnMatches[i][1];
            float distanceRatio = bestMatch.distance / betterMatch.distance;
            if (distanceRatio < minRatio)
            {
                m_Matches.push_back(bestMatch);
            }
        }
        
    }
    else
    {
        cout<<"Cross-Check"<<endl;
        Ptr<cv::DescriptorMatcher> BFMatcher(new cv::BFMatcher(cv::NORM_HAMMING, true));
        BFMatcher->match(queryDescriptors,trainDescriptors, m_Matches );
    }
    Mat homo;
    float homographyReprojectionThreshold = 1.0;
    bool homographyFound = refineMatchesWithHomography(
                                                       queryKeyPoints,trainKeyPoints,homographyReprojectionThreshold,m_Matches,homo);
    
    if (!homographyFound)
        return false;
    else
    {
        double h[9];
        bool isFound=true;
        h[0]=homo.at<double>(0,0);
        h[1]=homo.at<double>(0,1);
        h[2]=homo.at<double>(0,2);
        h[3]=homo.at<double>(1,0);
        h[4]=homo.at<double>(1,1);
        h[5]=homo.at<double>(1,2);
        h[6]=homo.at<double>(2,0);
        h[7]=homo.at<double>(2,1);
        h[8]=homo.at<double>(2,2);
        printf("homo\n"
               "%f %f %f\n"
               "%f %f %f\n"
               "%f %f %f\n",
               homo.at<double>(0,0), homo.at<double>(0,1), homo.at<double>(0,2),
               homo.at<double>(1,0), homo.at<double>(1,1), homo.at<double>(1,2),
               homo.at<double>(2,0), homo.at<double>(2,1), homo.at<double>(2,2));
        for (int i=0;i<4;i++)
        {
            float x = (float)srcCorner[i].x;
            float y = (float)srcCorner[i].y;
            float Z = (float)1./(h[6]*x + h[7]*y + h[8]);
            float X = (float)(h[0]*x + h[1]*y + h[2])*Z;
            float Y = (float)(h[3]*x + h[4]*y + h[5])*Z;
            //if(X>=0&&X<width&&Y>=0&&Y<height)
            {
                dstCorner[i]=Point(int(X),int(Y));
            }
            //else
            {
                //isFound &= false;
            }
        }
        if (isFound)
        {
            
            //抓到的特徵點畫上線條
            line(frameImg,dstCorner[0],dstCorner[1],CV_RGB(255,0,0),2);
            line(frameImg,dstCorner[1],dstCorner[2],CV_RGB(255,0,0),2);
            line(frameImg,dstCorner[2],dstCorner[3],CV_RGB(255,0,0),2);
            line(frameImg,dstCorner[3],dstCorner[0],CV_RGB(255,0,0),2);
            return true;
        }
        return true;
    }
    
    
}




int main()
{
//    string filename = "D:\\Img\\Heads\\03.jpg";
    src = imread(getmiku(),0);
    width = src.cols;
    height = src.rows;
    string detectorType = "ORB";
    string descriptorType = "ORB";
    string matcherType = "BruteForce";
    
    Ptr<FeatureDetector> featureDetector;
    Ptr<DescriptorExtractor> descriptorExtractor;
    Ptr<DescriptorMatcher> descriptorMatcher;
    if( !createDetectorDescriptorMatcher( detectorType, descriptorType, matcherType, featureDetector, descriptorExtractor, descriptorMatcher ) )
    {
        cout<<"Creat Detector Descriptor Matcher False!"<<endl;
        return -1;
    }
    //Intial: read the pattern img keyPoint
    vector<KeyPoint> queryKeypoints;
    Mat queryDescriptor;
    featureDetector->detect(src,queryKeypoints);
    descriptorExtractor->compute(src,queryKeypoints,queryDescriptor);
    
    
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
    {
        cout<<"Can't Open Camera!"<<endl;
        return -1;
    }
    srcCorner[0] = Point(0,0);
    srcCorner[1] = Point(width,0);
    srcCorner[2] = Point(width,height);
    srcCorner[3] = Point(0,height);
    
    vector<KeyPoint> trainKeypoints;
    Mat trainDescriptor;
    
    Mat frame,grayFrame;
    char key=0;
    
    //frame = imread("D:\\Img\\Heads\\00.jpg");
    
    while (key!=27)
    {
        cap>>frame;
        frame.copyTo(frameImg);
        cvtColor(frame,grayFrame,CV_BGR2GRAY);
        trainKeypoints.clear();
        // if (!trainDescriptor.empty())
        trainDescriptor.setTo(0);
        featureDetector->detect(grayFrame,trainKeypoints);
        
        if(trainKeypoints.size()!=0){
            descriptorExtractor->compute(grayFrame,trainKeypoints,trainDescriptor);
            bool isFound = matchingDescriptor(queryKeypoints,trainKeypoints,queryDescriptor,trainDescriptor,descriptorMatcher);
            
            imshow("foundImg",frameImg);
            
        }  
        key = waitKey(1);
        
    }  
    
}  




