#ifndef POISSON_BLENDER_H
#define POISSON_BLENDER_H



#include<opencv2/core/core.hpp>
#include<string>
#include<random>


class poisson_blender
{
private:


	//param
	double _size_rate;
	int _itr;
	double _th;

	cv::Mat Kh;
	cv::Mat Kv;
	cv::Mat K;

    int max_rand;

    std::default_random_engine rand_engine;
    std::uniform_int_distribution<size_t> rand_dist;
    
	void initK();
    void initRand();
    size_t get_rand();
    
    //rand choice the Rect to set object image (mask is 5 pixes bigger than object bounding)
	//cv::Rect choicePlace(const cv::Mat &bg, const cv::Mat &mask);

    void split_png(const cv::Mat &bgra, cv::Mat &bgr, cv::Mat &a);
    cv::Rect findMaskBoundingBox(const cv::Mat &bin_mask_8UC1);


    cv::Size get_fit_size(const cv::Size &bg, const cv::Size &mask_compact);


	bool isIntersect(const cv::Rect &region1,const cv::Rect &region2);
	//rand choice the Rect to set object image in the feasiblie_domain (mask is 5 pixes bigger than object bounding)
    bool choicePlace(const cv::Size &bg_size, const cv::Size &mask_size, std::vector<cv::Rect> &reject_region, cv::Rect &output);

    //region2 cover region1 area / region1.area
    double coverRate(const cv::Rect &region1, const cv::Rect &region2);
    //cover rate
    bool choicePlace2Cover(const cv::Size &bg_size, const cv::Size &mask_size, std::vector<cv::Rect> &reject_region, cv::Rect &output, double min_cover_rate, double max_cover_rate);

	//get the object bounding box
	//cv::Rect findObjectBoundingBox(const cv::Mat &Mask);

	//@bgr_image:  CV_8UC3 or CV_32FC3,BGR image
	//@Fh_out: CV_32FC3, grad x
	//@Fv_out: CV_32FC3, grad y
	void imgrad(const cv::Mat &bgr_image, cv::Mat &Fh_out, cv::Mat &Fv_out);

	//@mix_image: CV_8UC3 or CV_32FC3,BGR image with the object pasted on
	//@X: CV_32FC3, grad x
	//@Y: CV_32FC3, grad y
	//@Mask: CV_8UC1, value = 0 or 255
	void PoissonJacobiSam(cv::Mat &mix_image, const cv::Mat &X, const cv::Mat &Y, const cv::Mat &Mask, int itr = 1024, double th = 0.003);

	//void binMask(const cv::Mat &mask, cv::Mat &out_mask);

	//bool save(const cv::Mat &image,
	//	const std::string &filename, const std::string &path,
	//	std::vector<cv::Rect> &bndbox, const std::vector<std::string> &label);

	//bool blend_one(const cv::Mat &input_bg, const cv::Mat &input_object, const cv::Mat input_mask, cv::Mat &blended_img, cv::Rect &bounding_box, std::vector<cv::Rect> &reject_region);

    bool blend_one(const cv::Mat &input_bg, const cv::Mat &input_object,cv::Mat &blended_img, cv::Rect &bounding_box, std::vector<cv::Rect> &reject_region);



	//@input_object: CV_8UC3, BGR image
	//@input_bg: CV_8UC3, BGR image
	//@Mask: CV_8UC1, value = 0 or 255
	//@blended_img: output image CV_8UC3
	//@bounding_box: bounding box of object
	//bool blend_old(const cv::Mat &input_bg, const cv::Mat &input_object, const cv::Mat input_mask, cv::Mat &blended_img, cv::Rect &bounding_box);

public:

	//@size_rate: the object bounding box area / blackground area
	//@itr: iterations
	//@th: break threshold
	void setParam(double size_rate , int itr , double th);

	//@size_rate: the object bounding box area / blackground area
	//@itr: iterations
	//@th: break threshold
	poisson_blender(double size_rate = 1.0 / 25.0, int itr = 1024, double th = 0.003);

	~poisson_blender();


	//@input_object: CV_8UC3, BGR image
	//@input_bg: CV_8UC3, BGR image
	//@blended_img: output image CV_8UC3
	//@bounding_box: bounding box of object
	//it help make mask by "createMask(...)"
	//bool blend(const cv::Mat &input_object, const cv::Mat &input_bg, cv::Mat &blended_img, cv::Rect &bounding_box);

	//@input_bg: CV_8UC3, BGR image
	//@input_object: CV_8UC3, a set of BGR image
	//@Mask: CV_8UC1, value = 0 or 255, order on input_object
	//@blended_img: output image CV_8UC3
	//@bounding_box: bounding box of each object
	//@is_blended: flags of each object, true -> object is mix in bg, false -> otherwise
	//bool blend(const cv::Mat &input_bg, const std::vector<cv::Mat> &input_object,
	//	const std::vector<cv::Mat> &input_mask, cv::Mat &blended_img, 
	//	std::vector<cv::Rect> &bounding_box, std::vector<bool> &is_blended);

	//bool blend(const std::string &input_bg, const std::vector<std::string> &input_object, 
	//	const std::vector<std::string> &input_mask, cv::Mat &blended_img, 
	//	std::vector<cv::Rect> &bounding_box, std::vector<bool> &is_blended);

    bool blend(const cv::Mat &input_bg, const std::vector<cv::Mat> &input_object, cv::Mat &blended_img,
        std::vector<cv::Rect> &bounding_box, std::vector<bool> &is_blended);

    bool blend(const std::string &input_bg, const std::vector<std::string> &input_object, cv::Mat &blended_img,
        std::vector<cv::Rect> &bounding_box, std::vector<bool> &is_blended);

    static void blend_thread(const cv::Mat *input_bg, const std::vector<cv::Mat> *input_object, cv::Mat *blended_img,
        std::vector<cv::Rect> *bounding_box, std::vector<bool> *is_blended, poisson_blender * p_this, int * p_flag);

	//test
	//void imgrad(const cv::Mat &bgr_image, cv::Mat &Fh_out, cv::Mat &Fv_out);

	bool save(const cv::Mat &image,
		const std::string &filename, const std::string &path, const std::string &xml_path,
		std::vector<cv::Rect> &bndbox, const std::vector<std::string> &label);

	//bool isIntersect(const cv::Rect &region1, const cv::Rect &region2);

};





#endif // !POISSON_BLENDING_H
