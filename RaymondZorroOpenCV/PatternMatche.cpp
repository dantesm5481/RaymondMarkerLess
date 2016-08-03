//
//  PatternMatche.cpp
//  RaymondZorroOpenCV
//
//  Created by Raymond on 2016/8/2.
//  Copyright © 2016年 Raymond. All rights reserved.
//
#define ClusterNum 10

#include "PatternMatche.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


Mat DrawAndMatchKeypoints(const Mat& Img1,const Mat& Img2,const vector<KeyPoint>& Keypoints1,
                          const vector<KeyPoint>& Keypoints2,const Mat& Descriptors1,const Mat& Descriptors2)
{
    Mat keyP1,keyP2;
    drawKeypoints(Img1,Keypoints1,keyP1,Scalar::all(-1),0);
    drawKeypoints(Img2,Keypoints2,keyP2,Scalar::all(-1),0);
    
    
    //此處為原秀出Demo side步驟
    //    imshow("img1 keyPoints",keyP1);
    //    imshow("img2 keyPoints",keyP2);
    
    // BruteForce
    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( "BruteForce" );
    // DMatch?
    vector<DMatch> matches;
    descriptorMatcher->match( Descriptors1, Descriptors2, matches );
    Mat show;
    drawMatches(Img1,Keypoints1,Img2,Keypoints2,matches,show,Scalar::all(-1),CV_RGB(255,255,255),Mat(),4);
    
    // 匹配的模板 match
    //    imshow("match",show);
    
    
    
    return show;
}

//测试OpenCV：class BOWTrainer
Mat BOWKeams(const Mat& img, const vector<KeyPoint>& Keypoints,
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
    Scalar color[]={
        CV_RGB(255,255,255),
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
    //    imshow("KeyPoints Clusrtering",demoCluster);
    
    return demoCluster;
    
}
