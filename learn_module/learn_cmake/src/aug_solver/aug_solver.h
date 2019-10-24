#ifndef AUG_SOLVER_H
#define AUG_SOLVER_H

#include<opencv2/core/core.hpp>
#include<string>
//all function will renew the output memory 

//struct aug_operation
//{
//    std::string key_fun;
//    std::vector<std::vector<double>> param;
//};

struct ___aug_operation
{
    std::string opt_name;
    bool is_random_param;
    std::vector<double> param;
};

class aug_png_slover
{
private:
    std::vector<std::string> pre_step_obj;
    std::vector<std::string> pre_step_label;

    std::vector<std::string> now_step_obj;
    std::vector<std::string> now_step_label;

    std::string aug_config_file_path;
    std::vector<std::vector<___aug_operation>> operations;
    bool is_load_opt;

    bool checkOutOperation_one(const std::string &opt_str,___aug_operation &opt);
    bool checkOutOperation_line(const std::string &line, std::vector<___aug_operation> &opt_line);
   
    bool checkOperation(const std::vector<std::vector<___aug_operation>> &aug_opts);
    bool checkRandomParam(const std::vector<double> &param, double allow_min, double allow_max);
    bool checkParam(const std::vector<double> &param, double allow_min, double allow_max);
    bool checkOutOperation();
    bool checkOutPreStepList(int step_number, const std::string &save_root_path, std::vector<std::vector<std::string>> &list);

    int check_continue_step(const std::string &save_root_path);
    bool augment_step(int step_number, const  std::string &save_root_path);

    
    void get_rand(int n, double min, double max, std::vector<double> &output);
    void get_rand(const std::vector<double> n_min_max, std::vector<double> &output);

    bool augment_one_mat_obj(const std::string &obj, std::vector<cv::Mat> &aug_obj, const std::vector<___aug_operation> &step_opt);

    int get_step_aug_number_one_mat(const std::vector<___aug_operation > &aug_opts);

public:
    bool augment_continue(const  std::string &save_root_path);

    bool augment(const std::vector<std::vector<std::string>> &obj_path, const std::vector<std::string> &label, const  std::string &save_root_path);//build mask

    aug_png_slover(const std::string &_aug_config_path = "./aug_config/AUG_ALL");
    ~aug_png_slover();
    
    bool set_aug_config(const std::string &_aug_config_path);
    bool set_aug_config(const std::vector<std::vector<___aug_operation>>  &_aug_config);

    bool is_set_aug_config();
    std::vector<std::vector<___aug_operation>> getOperation();
    void printOperation(const std::vector<std::vector<___aug_operation>> &aug_opts);
    void printOperation();

    bool save_aug_config(const std::string &save_path);
    bool save_aug_config(const std::vector<std::vector<___aug_operation>> &aug_opts, const std::string &save_path);

};



#endif // !AUGMENTOR_H

