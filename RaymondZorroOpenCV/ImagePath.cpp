//
//  ImagePath.cpp
//  RaymondZorroOpenCV
//
//  Created by Raymond on 2016/8/1.
//  Copyright © 2016年 Raymond. All rights reserved.
//

#include "ImagePath.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

std::string getImgPath1(){return "/Users/dantesm5481/Desktop/runingproject/RaymondZorroOpenCV/pyrid1.jpg";};
std::string getImgPath2(){return "/Users/dantesm5481/Desktop/runingproject/RaymondZorroOpenCV/pyrid2.jpg";};

std::string getmiku(){return "/Users/dantesm5481/Desktop/runingproject/RaymondZorroOpenCV/miku.jpg";};

void imagecheck(Mat mat){
    (!mat.data)? std::cout << "Mat Wrong Path" <<endl :
    std::cout << " Mat load sucess" << std::endl;
}
