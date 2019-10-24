#ifndef AUG_FUNCTION_H
#define AUG_FUNCTION_H

#include<opencv2/core/core.hpp>
#include<vector>
#include<unordered_map>

enum IMAGE_POSITION
{
    UL = 1,
    U = 2,
    UR = 3,
    L = 4,

    R = 6,
    DL = 7,
    D = 8,
    DR = 9,

    UD = 28,
    LR = 46
};

enum AXIS
{
    X = 1,
    Y = 2
};

//all function will renew the output memory 
class augmentor_png
{
private:
    //ready to set private
    static void imCopy(const cv::Mat &src, std::vector<cv::Mat> &outputs, int n);
    static cv::Mat imRotate_one(const cv::Mat& img, double angle);
    static cv::Rect imCrop_rect(const cv::Size &img_size, double percentage);
    static cv::Mat imAffine_one(const cv::Mat& img, int XorY, double offset_rate);
    static cv::Mat imAddNoise_gauss_one(const cv::Mat &img, double noisiness);
    
    static cv::Mat imHue_one(const cv::Mat &img, double H_offset);
    static cv::Mat imLightness_one(const cv::Mat &img, double H_offset);
    static cv::Mat imSaturation_one(const cv::Mat &img, double H_offset);

    static cv::Mat imDistort_one(const cv::Mat &input, double rate);

    //type: 1 2 3
    //      4   6
    //      7 8 9
    static cv::Mat imPerspective_one(const cv::Mat& img, int type, double offset_rate);

    //random
    //static bool is_init_rand;
    //static void tryInitRand();
    //static int augRand();

    static int getAugRand(int min,int max);
    static void getAugRand(int n, int min, int max, std::vector<int> &rand_output);
    static double getAugRand(double min, double max);
    static void getAugRand(int n, double min, double max, std::vector<double> rand_output);
public:

    //平面旋转
    static void imRotate(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &angles);
  
    //镜像
    static void imFlip(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj);
    static void imFlipUD(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj);
    static void imFlipLR(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj);

    //裁剪
    static void imCrop(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &rate);
 
    //仿射
    static void imAffine(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &rates);
    static void imAffineX(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &rates);
    static void imAffineY(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &rates);


    //加噪
    static void imAddNoise_gauss(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &noisiness);
  
    //色相
    static void imHue(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &H_offset);
      
    //亮度
    static void imLightness(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &L_offset);

    //饱和度
    static void imSaturation(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &S_offset);

    //透视
    static void imPerspectiveUL(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj,  const std::vector<double> &rates);
    static void imPerspectiveU(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj,  const std::vector<double> &rates);
    static void imPerspectiveUR(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj,  const std::vector<double> &rates);
    static void imPerspectiveL(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj,  const std::vector<double> &rates);
    static void imPerspective(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj,  const std::vector<double> &rates);
    static void imPerspectiveR(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj,  const std::vector<double> &rates);
    static void imPerspectiveDL(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj,  const std::vector<double> &rates);
    static void imPerspectiveD(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj,  const std::vector<double> &rates);
    static void imPerspectiveDR(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj,  const std::vector<double> &rates);

    //扭曲
    static void imDistort(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &rates);

    //图像金字塔
    static void imPyramid(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj,  int down_number, int up_number);

};



#endif // !AUGMENTOR_H

