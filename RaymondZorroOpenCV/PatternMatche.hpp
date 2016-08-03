//
//  PatternMatche.hpp
//  RaymondZorroOpenCV
//
//  Created by Raymond on 2016/8/2.
//  Copyright © 2016年 Raymond. All rights reserved.
//

#ifndef PatternMatche_hpp
#define PatternMatche_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#endif /* PatternMatche_hpp */

Mat BOWKeams(const Mat& img, const vector<KeyPoint>& Keypoints,
             const Mat& Descriptors, Mat& centers);

Mat DrawAndMatchKeypoints(const Mat& Img1,const Mat& Img2,const vector<KeyPoint>& Keypoints1,
                          const vector<KeyPoint>& Keypoints2,const Mat& Descriptors1,const Mat& Descriptors2);