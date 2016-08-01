//此處為 使用到的功能都要進行 inculde
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>


#include "ImagePath.cpp";

using namespace cv;
using namespace std;

#define ClusterNum 10


// Load Image Path check point
void imagecheck(Mat mat){
    (!mat.data)? std::cout << "Mat Wrong Path" <<endl :
                 std::cout << " Mat load sucess" << std::endl;
    }

void DrawAndMatchKeypoints(const Mat& Img1,const Mat& Img2,const vector<KeyPoint>& Keypoints1,
                           const vector<KeyPoint>& Keypoints2,const Mat& Descriptors1,const Mat& Descriptors2);

Mat Cameraopen(Mat mat){
    VideoCapture cap(0);
    cap.read(mat);
    return mat;
}

void DrawAndMatchKeypoints(const Mat& Img1,const Mat& Img2,const vector<KeyPoint>& Keypoints1,
                           const vector<KeyPoint>& Keypoints2,const Mat& Descriptors1,const Mat& Descriptors2)
{
    Mat keyP1,keyP2;
    drawKeypoints(Img1,Keypoints1,keyP1,Scalar::all(-1),0);
    drawKeypoints(Img2,Keypoints2,keyP2,Scalar::all(-1),0);
//    putText(keyP1, "drawKeyPoints", cvPoint(10,30), FONT_HERSHEY_SIMPLEX, 1 ,Scalar :: all(-1));
//    putText(keyP2, "drawKeyPoints", cvPoint(10,30), FONT_HERSHEY_SIMPLEX, 1 ,Scalar :: all(-1));
    imshow("img1 keyPoints",keyP1);
    imshow("img2 keyPoints",keyP2);
    
    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( "BruteForce" );
    vector<DMatch> matches;
    descriptorMatcher->match( Descriptors1, Descriptors2, matches );
    Mat show;
    drawMatches(Img1,Keypoints1,Img2,Keypoints2,matches,show,Scalar::all(-1),CV_RGB(255,255,255),Mat(),4);
    putText(show, "drawMatchKeyPoints", cvPoint(10,30), FONT_HERSHEY_SIMPLEX, 1 ,Scalar :: all(-1));
    imshow("match",show);
}

//测试OpenCV：class BOWTrainer
void BOWKeams(const Mat& img, const vector<KeyPoint>& Keypoints,
              const Mat& Descriptors, Mat& centers)
{
    //BOW的kmeans算法聚类;
    BOWKMeansTrainer bowK(ClusterNum,
                          cvTermCriteria (CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 0.1),3,2);
    centers = bowK.cluster(Descriptors);
    cout<<endl<<"< cluster num: "<<centers.rows<<" > "<<endl;
    
    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( "BruteForce" );
    vector<DMatch> matches;
    descriptorMatcher->match(Descriptors,centers,matches);
    //const Mat& queryDescriptors, const Mat& trainDescriptors第一个参数是待分類節點，第二个参数是聚類中心;
    Mat demoCluster;
    img.copyTo(demoCluster);
    
    //為每一類keyPoint定義一種颜色
    Scalar color[]={CV_RGB(255,255,255),
        CV_RGB(255,0,0),CV_RGB(0,255,0),CV_RGB(0,0,255),
        CV_RGB(255,255,0),CV_RGB(255,0,255),CV_RGB(0,255,255),
        CV_RGB(123,123,0),CV_RGB(0,123,123),CV_RGB(123,0,123)};
    
    
    for (vector<DMatch>::iterator iter=matches.begin();iter!=matches.end();iter++)
    {
        cout<<"< descriptorsIdx:"<<iter->queryIdx<<"  centersIdx:  "<<iter->trainIdx
        <<" distincs:"<<iter->distance<<" >"<<endl;
        Point center= Keypoints[iter->queryIdx].pt;
        circle(demoCluster,center,2,color[iter->trainIdx],-1);
    }
    putText(demoCluster, "KeyPoints Clustering: 一種顏色代表一種類別",
            cvPoint(10,30), FONT_HERSHEY_SIMPLEX, 1 ,Scalar :: all(-1));
    imshow("KeyPoints Clusrtering",demoCluster);
    
}



//開啟 OpenCV Camera
int main(){
    
    
    cv::initModule_nonfree();//使用SIFT/SURF create之前，必須先initModule_<modulename>();
    
    cout << "< Creating detector, descriptor extractor and descriptor matcher ...";
    Ptr<FeatureDetector> detector = FeatureDetector::create( "SIFT" );
    
    Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create( "SIFT" );
    
    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( "BruteForce" );
    
    
    
    cout << ">" << endl;
    
    if( detector.empty() || descriptorExtractor.empty() )
    {
        cout << "Can not create detector or descriptor exstractor or descriptor matcher of given types" << endl;
        return -1;
    }
    cout << endl << "< Reading images..." << endl;
    Mat img1 = imread("/Users/dantesm5481/Desktop/runingproject/RaymondZorroOpenCV/pyrid1.jpg");
    Mat img2 = imread("/Users/dantesm5481/Desktop/runingproject/RaymondZorroOpenCV/pyrid2.jpg");
//    Mat img1 = imread("/Users/dantesm5481/Desktop/OpenCV/miku.jpg");
//    Mat img2 = imread("/Users/dantesm5481/Desktop/OpenCV/miku.jpg");
    
    cout<<endl<<">"<<endl;
    
    
    //detect keypoints;
    cout << endl << "< Extracting keypoints from images..." << endl;
    vector<KeyPoint> keypoints1,keypoints2;
    detector->detect( img1, keypoints1 );
    detector->detect( img2, keypoints2 );
    cout <<"img1:"<< keypoints1.size() << " points  img2:" <<keypoints2.size()
    << " points" << endl << ">" << endl;
    
    //compute descriptors for keypoints;
    cout << "< Computing descriptors for keypoints from images..." << endl;
    Mat descriptors1,descriptors2;
    descriptorExtractor->compute( img1, keypoints1, descriptors1 );
    descriptorExtractor->compute( img2, keypoints2, descriptors2 );
    
    
    cout<<endl<<"< Descriptoers Size: "<<descriptors2.size()<<" >"<<endl;
    cout<<endl<<"descriptor's col: "<<descriptors2.cols<<endl
    <<"descriptor's row: "<<descriptors2.rows<<endl;
    cout << ">" << endl;
    
    //Draw And Match img1,img2 keypoints
    //匹配的過程是退特徵點的的descriptors進行match;
    DrawAndMatchKeypoints(img1,img2,keypoints1,keypoints2,descriptors1,descriptors2);
    
    Mat center;
    //對img1提取特徵點，並且聚類
    //測試OpenCV：class BOWTrainer
    BOWKeams(img1,keypoints1,descriptors1,center);
    
    
    waitKey(0);
    
    
    
    

    
    return 0;
    
}








