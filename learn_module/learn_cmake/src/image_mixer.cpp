// image_mixer.cpp : 定义控制台应用程序的入口点。
//
#include"poisson_blending/poisson_blender.h"

#include"stl_expand/stl_expand.hpp"

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>

#include<iostream>
#include<iomanip>
#include<ctime>
#include<string>
#include<fstream>
#include<random>

#include<thread>
//#include <omp.h>

#include <boost/program_options.hpp>  
namespace po = boost::program_options;
using namespace std;
using namespace cv;

string getToday()
{
    static string day_string;
    if (day_string != "") return day_string;
    time_t _time_t = time(NULL);
    tm ttt;
//windows
    //localtime_s(&ttt, &_time_t);
//unix
    tm *p_tt=localtime(&_time_t);
    ttt = *p_tt;
    delete p_tt;
//////////////////

    day_string = to_string(ttt.tm_year + 1900)+"-";
    if (ttt.tm_mon + 1 <= 9) day_string += "0";
    day_string += to_string(ttt.tm_mon + 1)+"-";
    if (ttt.tm_mday <= 9) day_string += "0";
    day_string += to_string(ttt.tm_mday);
    return day_string;
}


bool checkOutAppointList(const string &list_path, vector<string> &bg, vector<vector<vector<string>>> &objs)
{
    ifstream list_file(list_path);
    if (!list_file.is_open()) return false;
    bg.resize(0);
    objs.resize(0);
    while (!list_file.eof())
    {
        std::string line;
        std::getline(list_file, line);
        vector<string> part;

        std::expand::split(line, " " , part);

        if (part.size() < 3 || part.size() % 2 != 1) continue;

        bg.push_back(part[0]);

        vector<vector<string>> one_line_obj(2);
        size_t m = (part.size() - 1) / 2;

        for(int i = 1;i<part.size();++i)
        {
            one_line_obj[(i - 1) % 2].push_back(part[i]);
        }
        objs.push_back(one_line_obj);
    }
    list_file.close();
    list_file.clear();
    return true;
}


void rand_combin(const vector<string> &bg_list, vector<string> obj_list,
  vector<string> label_list, int n, int mix_obj,int max_obj,
    vector<string> &bg, vector<vector<vector<string>>> &objs,const string &save_path)
{
    std::expand::create_directory(save_path);
    ofstream rand_combin_list_file(save_path+"/rand_combin_config.txt");
    //if (!rand_combin_config_file.is_open()) return;

    std::default_random_engine bg_e(time(0));
    std::uniform_int_distribution<size_t> bg_rand(0, bg_list.size() - 1);

    std::default_random_engine obj_e(time(0));
    std::uniform_int_distribution<size_t> obj_rand(0, obj_list.size() - 1);

    std::default_random_engine obj_n_e(time(0));
    std::uniform_int_distribution<size_t> obj_n_rand(mix_obj, max_obj);

    //Shuffle 
    for (int i = 0; i < obj_list.size(); ++i)
    {
        size_t kkk = obj_rand(obj_e);
        string t;
        {t = obj_list[i];  obj_list[i] = obj_list[kkk]; obj_list[kkk] = t; }
        {t = label_list[i];  label_list[i] = label_list[kkk]; label_list[kkk] = t;}
    }

    int obj_index = 0;
    int obj_size = obj_list.size();

    bg.resize(n);
    objs.resize(n);

    for (int i = 0; i < n; ++i)
    {
        size_t bg_index = bg_rand(bg_e);
        rand_combin_list_file << bg_list[bg_index];
        bg[i] = bg_list[bg_index];

        objs[i].resize(2);
        size_t m = obj_n_rand(obj_n_e);
        objs[i][0].resize(m);
        objs[i][1].resize(m);

        int j = 0;
        for (; obj_index != obj_size&&j < m; ++j, ++obj_index)
        {
            rand_combin_list_file << " " << obj_list[obj_index];
            rand_combin_list_file << " " << label_list[obj_index];
            objs[i][0][j] = obj_list[obj_index];
            objs[i][1][j] = label_list[obj_index];
        }
        for (; j < m; ++j)
        {
            size_t kkk = obj_rand(obj_e);
            rand_combin_list_file << " " << obj_list[kkk];
            rand_combin_list_file << " " << label_list[kkk];
            objs[i][0][j] = obj_list[kkk];
            objs[i][1][j] = label_list[kkk];
        }
        rand_combin_list_file << endl;
    }
    rand_combin_list_file.close();
    rand_combin_list_file.clear();
    return;
}

void rand_combin_two_same(const vector<string> &bg_list, vector<string> obj_list,
    vector<string> label_list, int n, int mix_obj, int max_obj,
    vector<string> &bg, vector<vector<vector<string>>> &objs, const string &save_path)
{
    std::expand::create_directory(save_path);
    ofstream rand_combin_list_file(save_path + "/rand_combin_config.txt");
    //if (!rand_combin_config_file.is_open()) return;

    std::default_random_engine bg_e(time(0));
    std::uniform_int_distribution<size_t> bg_rand(0, bg_list.size() - 1);

    std::default_random_engine obj_e(time(0));
    std::uniform_int_distribution<size_t> obj_rand(0, obj_list.size() - 1);

    std::default_random_engine obj_n_e(time(0));
    std::uniform_int_distribution<size_t> obj_n_rand(mix_obj, max_obj);

    //Shuffle 
    for (int i = 0; i < obj_list.size(); ++i)
    {
        size_t kkk = obj_rand(obj_e);
        string t;
        {t = obj_list[i];  obj_list[i] = obj_list[kkk]; obj_list[kkk] = t; }
        {t = label_list[i];  label_list[i] = label_list[kkk]; label_list[kkk] = t; }
    }

    int obj_index = 0;
    int obj_size = obj_list.size();

    bg.resize(n);
    objs.resize(n);

    for (int i = 0; i < n; ++i)
    {
        size_t bg_index = bg_rand(bg_e);
        rand_combin_list_file << bg_list[bg_index];
        bg[i] = bg_list[bg_index];

        objs[i].resize(2);
        size_t m = obj_n_rand(obj_n_e);
        objs[i][0].resize(m);
        objs[i][1].resize(m);

        int j = 0;
        for (; obj_index != obj_size&&j < m; ++j, ++obj_index)
        {
            if (j % 2 == 1)
            {
                while (1)
                {
                    size_t kkk = obj_rand(obj_e);
                    if (label_list[kkk] == objs[i][1][j - 1])
                    {
                        rand_combin_list_file << " " << obj_list[kkk];
                        rand_combin_list_file << " " << label_list[kkk];
                        objs[i][0][j] = obj_list[kkk];
                        objs[i][1][j] = label_list[kkk];
                        break;
                    }
                    //cout << label_list[kkk] << "    "<<objs[i][1][j - 1] << endl;
                }
                continue;
            }
            rand_combin_list_file << " " << obj_list[obj_index];
            rand_combin_list_file << " " << label_list[obj_index];
            objs[i][0][j] = obj_list[obj_index];
            objs[i][1][j] = label_list[obj_index];
        }
        for (; j < m; ++j)
        {
            if (j % 2 == 1)
            {
                while (1)
                {
                    size_t kkk = obj_rand(obj_e);
                    if (label_list[kkk] == objs[i][1][j - 1])
                    {
                        rand_combin_list_file << " " << obj_list[kkk];
                        rand_combin_list_file << " " << label_list[kkk];
                        objs[i][0][j] = obj_list[kkk];
                        objs[i][1][j] = label_list[kkk];
                        break;
                    }
                    //cout << label_list[kkk] << "    " << objs[i][1][j - 1] << endl;
                }
                continue;
            }
            size_t kkk = obj_rand(obj_e);
            rand_combin_list_file << " " << obj_list[kkk];
            rand_combin_list_file << " " << label_list[kkk];
            objs[i][0][j] = obj_list[kkk];
            objs[i][1][j] = label_list[kkk];
        }
        rand_combin_list_file << endl;
    }
    rand_combin_list_file.close();
    rand_combin_list_file.clear();
    return;
}

void rand_combin_two_different(const vector<string> &bg_list, vector<string> obj_list,
    vector<string> label_list, int n, int mix_obj, int max_obj,
    vector<string> &bg, vector<vector<vector<string>>> &objs, const string &save_path)
{
    std::expand::create_directory(save_path);
    ofstream rand_combin_list_file(save_path + "/rand_combin_config.txt");
    //if (!rand_combin_config_file.is_open()) return;

    std::default_random_engine bg_e(time(0));
    std::uniform_int_distribution<size_t> bg_rand(0, bg_list.size() - 1);

    std::default_random_engine obj_e(time(0));
    std::uniform_int_distribution<size_t> obj_rand(0, obj_list.size() - 1);

    std::default_random_engine obj_n_e(time(0));
    std::uniform_int_distribution<size_t> obj_n_rand(mix_obj, max_obj);

    //Shuffle 
    for (int i = 0; i < obj_list.size(); ++i)
    {
        size_t kkk = obj_rand(obj_e);
        string t;
        {t = obj_list[i];  obj_list[i] = obj_list[kkk]; obj_list[kkk] = t; }
        {t = label_list[i];  label_list[i] = label_list[kkk]; label_list[kkk] = t; }
    }

    int obj_index = 0;
    int obj_size = obj_list.size();

    bg.resize(n);
    objs.resize(n);

    for (int i = 0; i < n; ++i)
    {
        size_t bg_index = bg_rand(bg_e);
        rand_combin_list_file << bg_list[bg_index];
        bg[i] = bg_list[bg_index];

        objs[i].resize(2);
        size_t m = obj_n_rand(obj_n_e);
        objs[i][0].resize(m);
        objs[i][1].resize(m);

        int j = 0;
        for (; obj_index != obj_size&&j < m; ++j, ++obj_index)
        {
            if (j % 2 == 1)
            {
                while (1)
                {
                    size_t kkk = obj_rand(obj_e);
                    if (label_list[kkk] != objs[i][1][j - 1])
                    {
                        rand_combin_list_file << " " << obj_list[kkk];
                        rand_combin_list_file << " " << label_list[kkk];
                        objs[i][0][j] = obj_list[kkk];
                        objs[i][1][j] = label_list[kkk];
                        break;
                    }
                    //cout << label_list[kkk] << "    "<<objs[i][1][j - 1] << endl;
                }
                continue;
            }
            rand_combin_list_file << " " << obj_list[obj_index];
            rand_combin_list_file << " " << label_list[obj_index];
            objs[i][0][j] = obj_list[obj_index];
            objs[i][1][j] = label_list[obj_index];
        }
        for (; j < m; ++j)
        {
            if (j % 2 == 1)
            {
                while (1)
                {
                    size_t kkk = obj_rand(obj_e);
                    if (label_list[kkk] != objs[i][1][j - 1])
                    {
                        rand_combin_list_file << " " << obj_list[kkk];
                        rand_combin_list_file << " " << label_list[kkk];
                        objs[i][0][j] = obj_list[kkk];
                        objs[i][1][j] = label_list[kkk];
                        break;
                    }
                    //cout << label_list[kkk] << "    " << objs[i][1][j - 1] << endl;
                }
                continue;
            }
            size_t kkk = obj_rand(obj_e);
            rand_combin_list_file << " " << obj_list[kkk];
            rand_combin_list_file << " " << label_list[kkk];
            objs[i][0][j] = obj_list[kkk];
            objs[i][1][j] = label_list[kkk];
        }
        rand_combin_list_file << endl;
    }
    rand_combin_list_file.close();
    rand_combin_list_file.clear();
    return;
}

void rand_combin_two(const vector<string> &bg_list, vector<string> obj_list,
    vector<string> label_list, int n, int mix_obj, int max_obj,
    vector<string> &bg, vector<vector<vector<string>>> &objs, const string &save_path)
{
    std::expand::create_directory(save_path);
    ofstream rand_combin_list_file(save_path + "/rand_combin_config.txt");
    //if (!rand_combin_config_file.is_open()) return;

    std::default_random_engine bg_e(time(0));
    std::uniform_int_distribution<size_t> bg_rand(0, bg_list.size() - 1);

    std::default_random_engine obj_e(time(0));
    std::uniform_int_distribution<size_t> obj_rand(0, obj_list.size() - 1);

    std::default_random_engine obj_n_e(time(0));
    std::uniform_int_distribution<size_t> obj_n_rand(mix_obj, max_obj);


    std::default_random_engine rand_0_1_engine(time(0));
    std::uniform_int_distribution<size_t> rand_0_1_dist(0, 1);


    //Shuffle 
    for (int i = 0; i < obj_list.size(); ++i)
    {
        size_t kkk = obj_rand(obj_e);
        string t;
        {t = obj_list[i];  obj_list[i] = obj_list[kkk]; obj_list[kkk] = t; }
        {t = label_list[i];  label_list[i] = label_list[kkk]; label_list[kkk] = t; }
    }

    int obj_index = 0;
    int obj_size = obj_list.size();

    bg.resize(n);
    objs.resize(n);

    for (int i = 0; i < n; ++i)
    {
        size_t bg_index = bg_rand(bg_e);
        rand_combin_list_file << bg_list[bg_index];
        bg[i] = bg_list[bg_index];

        objs[i].resize(2);
        size_t m = obj_n_rand(obj_n_e);
        objs[i][0].resize(m);
        objs[i][1].resize(m);

        int j = 0;
        for (; obj_index != obj_size&&j < m; ++j, ++obj_index)
        {
            if (j % 2 == 1)
            {
                if (rand_0_1_dist(rand_0_1_engine))
                {
                    while (1)
                    {
                        size_t kkk = obj_rand(obj_e);
                        if (label_list[kkk] == objs[i][1][j - 1])
                        {
                            rand_combin_list_file << " " << obj_list[kkk];
                            rand_combin_list_file << " " << label_list[kkk];
                            objs[i][0][j] = obj_list[kkk];
                            objs[i][1][j] = label_list[kkk];
                            break;
                        }
                        //cout << label_list[kkk] << "    "<<objs[i][1][j - 1] << endl;
                    }
                }
                else
                {
                    while (1)
                    {
                        size_t kkk = obj_rand(obj_e);
                        if (label_list[kkk] != objs[i][1][j - 1])
                        {
                            rand_combin_list_file << " " << obj_list[kkk];
                            rand_combin_list_file << " " << label_list[kkk];
                            objs[i][0][j] = obj_list[kkk];
                            objs[i][1][j] = label_list[kkk];
                            break;
                        }
                        //cout << label_list[kkk] << "    "<<objs[i][1][j - 1] << endl;
                    }
                }
                continue;
            }
            rand_combin_list_file << " " << obj_list[obj_index];
            rand_combin_list_file << " " << label_list[obj_index];
            objs[i][0][j] = obj_list[obj_index];
            objs[i][1][j] = label_list[obj_index];
        }
        for (; j < m; ++j)
        {
            if (j % 2 == 1)
            {
                if (rand_0_1_dist(rand_0_1_engine))
                {
                    while (1)
                    {
                        size_t kkk = obj_rand(obj_e);
                        if (label_list[kkk] == objs[i][1][j - 1])
                        {
                            rand_combin_list_file << " " << obj_list[kkk];
                            rand_combin_list_file << " " << label_list[kkk];
                            objs[i][0][j] = obj_list[kkk];
                            objs[i][1][j] = label_list[kkk];
                            break;
                        }
                        //cout << label_list[kkk] << "    "<<objs[i][1][j - 1] << endl;
                    }
                }
                else
                {
                    while (1)
                    {
                        size_t kkk = obj_rand(obj_e);
                        if (label_list[kkk] != objs[i][1][j - 1])
                        {
                            rand_combin_list_file << " " << obj_list[kkk];
                            rand_combin_list_file << " " << label_list[kkk];
                            objs[i][0][j] = obj_list[kkk];
                            objs[i][1][j] = label_list[kkk];
                            break;
                        }
                        //cout << label_list[kkk] << "    "<<objs[i][1][j - 1] << endl;
                    }
                }
                continue;
            }
            size_t kkk = obj_rand(obj_e);
            rand_combin_list_file << " " << obj_list[kkk];
            rand_combin_list_file << " " << label_list[kkk];
            objs[i][0][j] = obj_list[kkk];
            objs[i][1][j] = label_list[kkk];
        }
        rand_combin_list_file << endl;
    }
    rand_combin_list_file.close();
    rand_combin_list_file.clear();
    return;
}



bool checkOutObjList(const string &list, vector<string> &obj_list,vector<string> &label_list)
{
    vector<vector<string>> parts;
    if (!std::expand::check_out_list(list, " ", parts)) { return false; }
    if (parts.size() != 2) { return false; }
    obj_list = parts[0];
    label_list = parts[1];
    return true;
}

bool checkOutBgList(const string &bg_list_path, vector<string> &bg_list)
{
    ifstream bg_file(bg_list_path);
    if (!bg_file.is_open())return false;
    while (!bg_file.eof())
    {
        string one_str;
        bg_file >> one_str;
        if (one_str.size() != 0)
            bg_list.push_back(one_str);
    }
    //cout << "read " << bg_list.size() << " bgs" << endl;
    return true;
}

//#include <boost/filesystem.hpp>  
//#include<boost/algorithm/string.hpp>
//namespace fs = boost::filesystem;
//
//bool get_image_names(const std::string& dir, std::vector<std::string>& filenames)
//{
//    fs::path path(dir);
//    if (!fs::exists(path))
//    {
//        return false;
//    }
//
//    fs::directory_iterator end_iter;
//    filenames.resize(0);
//    for (fs::directory_iterator iter(path); iter != end_iter; ++iter)
//    {
//        if (fs::is_regular_file(iter->status()))
//        {
//            std::string ext = iter->path().extension().string();
//            boost::algorithm::to_lower(ext);
//            if (ext == ".jpg"
//                || ext == ".jpeg"
//                || ext == ".gif"
//                || ext == ".png"
//                || ext == ".bmp")
//                filenames.push_back(iter->path().string());
//        }
//
//        //if (fs::is_directory(iter->status()))
//        //{
//        //    get_filenames(iter->path().string(), filenames);
//        //}
//    }
//
//    return true;
//}


void th_fun(poisson_blender *pd,
    int i,
    const vector<string> *bg,
    const vector<vector<vector<string>>> *obj_mask_label,
    ofstream *mix_msg_file,
    const string *save_image_path,
    const string *save_xml_path
    )
{
    Mat mix_img;
    vector<Rect> bd_set;
    vector<bool> is_blended;

    bool is_blend;
    bool is_save;

    


    is_blend = (*pd).blend((*bg)[i], (*obj_mask_label)[i][0], mix_img, bd_set, is_blended);


    //clock_t begin, end;
    //begin = clock();

    string xml_name;
    xml_name = "fusion";
    xml_name += "-" + getToday();
    xml_name += "-" + to_string(i + 1+10000);
    is_save = (*pd).save(mix_img, xml_name, (*save_image_path), (*save_xml_path), bd_set, (*obj_mask_label)[i][1]);

    string out_msg;
    out_msg = to_string(i + 1) + "    ";
    if (is_blend) out_msg += "blend = succeed    ";
    else
    {
        out_msg += "blend = fail in the index of ";
        out_msg += to_string(expand::find(is_blended, false));
        out_msg += "    ";
    }
    if (is_save) out_msg += "save = succeed    ";
    else out_msg += "save = fail    ";

    cout << out_msg << endl;
    (*mix_msg_file) << out_msg << endl;

    //end = clock();
    //std::cout <<"save = "<< double(end - begin) / CLOCKS_PER_SEC * 1000 << "ms    ";

    return;
}

int test_parallel_for(int &a)
{
    for (int i = 0; i < 1000000; i++)
    {
        if(i%2==0)
            a = sqrt(a);
        else
        {
            a = a*a + a;
        }
    }

    return a;
}


//////////////////////////////////////////////
//global
//poisson_blender g_pd;
//vector<string> g_bg;
//vector<vector<vector<string>>> g_obj_mask_label;
//ofstream g_mix_msg_file("./image_mixer_log/mixer_log.txt");
////ofstream mix_msg_file("./image_mixer_log/mixer_log.txt");
//string g_save_image_path;
//string g_save_xml_path;
//
//void g_th_fun(int i)
//{
//    Mat mix_img;
//    vector<Rect> bd_set;
//    vector<bool> is_blended;
//
//    bool is_blend;
//    bool is_save;
//
//    is_blend = g_pd.blend(g_bg[i], g_obj_mask_label[i][0], g_obj_mask_label[i][1], mix_img, bd_set, is_blended);
//
//    string xml_name;
//    xml_name = "fusion";
//    xml_name += "-" + getToday();
//    xml_name += "-" + to_string(i + 1);
//    is_save = g_pd.save(mix_img, xml_name, g_save_image_path, g_save_xml_path, bd_set, g_obj_mask_label[i][2]);
//
//    string out_msg;
//    out_msg = to_string(i + 1) + "    ";
//    if (is_blend) out_msg += "blend = succeed    ";
//    else
//    {
//        out_msg += "blend = fail in the index of ";
//        out_msg += to_string(expand::find(is_blended, false));
//        out_msg += "    ";
//    }
//    if (is_save) out_msg += "save = succeed    ";
//    else out_msg += "save = fail    ";
//
//    cout << out_msg << endl;;
//    g_mix_msg_file << out_msg << endl;
//    return;
//}


//////////////////////////////////////////////

bool load_img(const string &bg_str, const vector<string> &obj_mask_labels_str, cv::Mat &bg, vector<cv::Mat> &obj_mask_labels)
{
    bg = cv::imread(bg_str);
    if (bg.empty()) { return false; }
    cv::Mat t;
    obj_mask_labels.resize(obj_mask_labels_str.size());
    for (int i = 0; i < obj_mask_labels_str.size(); ++i)
    {
        t = cv::imread(obj_mask_labels_str[i], CV_LOAD_IMAGE_UNCHANGED);
        if (t.empty() || t.channels() != 4) { return false; }
        t.copyTo(obj_mask_labels[i]);
    }
    return true;
}



int main(int argc, char **argv)
{
    //ifstream txt("./list_no_error.txt");
    //ofstream otxt("./real_list.txt");
    //while (!txt.eof())
    //{
    //    string line;
    //    getline(txt, line);
    //    if (line.size()!= 0)
    //    {
    //        otxt << "Z:\\" << line.substr(12) << endl;

    //    }
    //}
    //txt.close();
    //txt.clear();
    //otxt.close();
    //otxt.clear();
    //cout << "fin" << endl;
    //system("pause");



    //cv::Mat yousuan_bgra = cv::imread("./ver3.0.0/aug/baishi/step4/obj_24.png",CV_LOAD_IMAGE_UNCHANGED);
    //std::vector<cv::Mat> v_bgra;
    //cv::Mat yousuan_a,yousuan_bgr;
    //cv::split(yousuan_bgra, v_bgra);
    //yousuan_a = v_bgra[3].clone();
    //v_bgra.pop_back();
    //cv::merge(v_bgra, yousuan_bgr);
    //for (int i = 0; i < yousuan_bgr.rows; ++i)
    //{
    //    cv::Vec3b * p_row_yousuan_bgr = yousuan_bgr.ptr<cv::Vec3b>(i);
    //    uchar *p_row_yousuan_a = yousuan_a.ptr<uchar>(i);
    //    for (int j = 0; j < yousuan_bgr.cols; ++j)
    //    {
    //        if (p_row_yousuan_a[j] != 255)
    //        {
    //            p_row_yousuan_bgr[j] = Vec3b(0, 255, 0);
    //        }
    //    }
    //}
    //cv::imwrite("yousuan_bgr.jpg", yousuan_bgr);
    //waitKey(0);
    //getchar();

//    clock_t start_para, finish_para;
//    double duration_para;
//    int a = 0;
//
//    cout << "start" << endl;
//    getchar();
//    start_para = clock();
//    
//    for (int i = 0; i<800; i++)
//        a = test_parallel_for(a);
//    finish_para = clock();
//    duration_para = (double)(finish_para - start_para) / CLOCKS_PER_SEC;
//    cout << duration_para << "s" << endl;
//
//
//    getchar();
//    start_para = clock();
//#pragma omp parallel for num_threads(4)
//    for (int i = 0; i<800; i++)
//        a = test_parallel_for(a);
//    finish_para = clock();
//    duration_para = (double)(finish_para - start_para) / CLOCKS_PER_SEC;
//    cout << duration_para << "s" << endl;
//
//    getchar();


    string bg_list_path;
    string bg_dir;
    string obj_list_path;
    int mix_number = 0;
    int min_mix_obj;
    int max_mix_obj;
    int index_param;

    po::options_description random_combine("RANDOM_COMBINE");
    random_combine.add_options()
        ("bg_list", po::value<string>(&bg_list_path), "The bg list file.")
        ("bg_dir", po::value<string>(&bg_dir), "The bg dir instead of bg list file.")
        ("obj_list", po::value<string>(&obj_list_path), "The obj list file, [obj] [mask] [label] in each line.")
        ("mix_number", po::value<int>(&mix_number), "The mix number.")
        ("min_mix_obj", po::value<int>(&min_mix_obj)->default_value(4), "The min number of obj in one bg.")
        ("max_mix_obj", po::value<int>(&max_mix_obj)->default_value(6), "The max number of obj in one bg.")
        ("index", po::value<int>(&index_param)->default_value(10000), "The first mix image's index.");

    string all_list_path;
    int begin_line;
    int end_line;
    po::options_description appoint_combine("APPOINT_COMBINE");
    appoint_combine.add_options()
        ("all_list", po::value<string>(&all_list_path), "list file, [bg] [obj1] [mask1] [label1] [obj2] [mask2] [label2] ... in each line.")
        ("begin_line", po::value<int>(&begin_line)->default_value(1), "begin from line [begin_line], helpful if unexpectedly interrupt")
        ("end_line", po::value<int>(&end_line)->default_value(INT_MAX), "end line, helpful if unexpectedly interrupt");

    string mode;
    string save_path;

    po::options_description all("config");
    all.add_options()
        ("mode", po::value<string>(&mode)->default_value("RANDOM_COMBINE"), "RANDOM_COMBINE  APPOINT_COMBINE.")
        ("save", po::value<string>(&save_path)->default_value("mix"), "The path for output.");
    
    all.add(random_combine).add(appoint_combine);

    if (argc == 1)
    {
        cout << all << endl;
        return -1;
    }

    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);
    }
    catch (boost::program_options::error_with_no_option_name &ex)
    {
        cout << ex.what() << endl;
        return -1;
    }
    
    //////////////////////////////////////////////////
    //bg_dir = "D:\\project\\ImAugmentor\\build\\bin\\release_x64\\ver3.0.3\\bg";
    //obj_list_path = "D:\\project\\ImAugmentor\\build\\bin\\release_x64\\ver3.0.3\\aug\\list.txt";
    //mix_number = 10000;
    //save_path = "D:\\project\\ImAugmentor\\build\\bin\\release_x64\\ver3.0.3\\mix";
    //mode = "APPOINT_COMBINE";
    //all_list_path = "./ver2.0.2/image_mixer_log/rand_combin_config.txt";

    //bg_dir = "./ver3.0.0/bg";
    //obj_list_path = "./ver3.0.0/list.txt";
    //mix_number = 20;
    //save_image_path = "./ver3.0.0/mix";

    //mode = "APPOINT_COMBINE";
    //save_path = "F:\\test_mix_time";
    //all_list_path = "G:\\ver3.0.2\\rand_combin_config_ip106.txt";
    //begin_line = 1;
    //end_line = 30;

    //save_path = "G:\\share\\ver3.0.2\\test2";
    //bg_dir = "G:\\share\\ver3.0.2\\bg";
    ////bg_list_path = "./bg_list.txt";
    //obj_list_path = "G:\\share\\ver3.0.2\\test_no\\list.txt";
    //mix_number = 30;
    //min_mix_obj = 6;
    //index_param = 60000;

    //////////////////////////////////////////////////

    if (mode == "RANDOM_COMBINE")
    {
        if ((bg_list_path.size() == 0 && bg_dir.size() == 0) || obj_list_path.size() == 0 || mix_number <= 0)
        {
            if (mix_number <= 0)
            {
                cout << "error with mix_number <= 0." << endl;
            }
            else
            {
                cout << all << endl;
            }
            return -1;
        }
    }
    else if (mode == "APPOINT_COMBINE")
    {
        if (all_list_path.size() == 0)
        {
            cout << "error: all_list = " << all_list_path << endl;
            return -1;
        }
    }
    else
    {
        cout << "mode error." << endl;
        return -1;
    }

    vector<string> bg;
    vector<vector<vector<string>>> obj_mask_label;

    if (mode == "RANDOM_COMBINE"&& (bg_list_path.size() != 0|| bg_dir.size() != 0) && obj_list_path.size() != 0 && mix_number > 0)
    {
        vector<string> bg_list;
        vector<string> obj_list;
        vector<string> label_list;
        if (!checkOutBgList(bg_list_path, bg_list))
        {
            std::vector<std::vector<std::string>> _temp_scan_file;
            if (!std::expand::scan_file_list(bg_dir, ".png.jpg.jpeg.gif.bmp", bg_list, std::expand::SCAN_ROOT_PATH_FILE_FULL_NAME,_temp_scan_file))
            {
                cout << "error: can not scan any bgs in, " << bg_dir << endl;;
                return -1;
            }
            else if(bg_list.size()==0)
            {
                cout << "error: can not scan any bgs in, " << bg_dir << endl;;
                return -1;
            }
        }
        if(!checkOutObjList(obj_list_path, obj_list, label_list))
        {
            cout << "error: can not read list, " << obj_list_path << endl;
            return -1;
        }

        //TODO:
        //rand_combin(bg_list, obj_list, label_list, mix_number, min_mix_obj, max_mix_obj, bg, obj_mask_label,save_path);
        rand_combin_two(bg_list, obj_list, label_list, mix_number, min_mix_obj, max_mix_obj, bg, obj_mask_label, save_path);
        //rand_combin_two_same(bg_list, obj_list, label_list, mix_number, min_mix_obj, max_mix_obj, bg, obj_mask_label, save_path);
        //rand_combin_two_different(bg_list, obj_list, label_list, mix_number, min_mix_obj, max_mix_obj, bg, obj_mask_label, save_path);
        begin_line = 1;

    }
    else if (mode == "APPOINT_COMBINE")
    {
        if (!checkOutAppointList(all_list_path, bg, obj_mask_label))
        {
            cout << "error: can not read list, " << all_list_path << endl;
            return -1;
        }
        mix_number = bg.size();
    }
    else
    {
        cout << all << endl;
        return -1;
    }

    if (!std::expand::create_directory(save_path))
    {
        cout << "Can create the output directory." << endl;
        return -1;
    }
    ofstream mix_msg_file(save_path + "/mixer_log.txt");

    poisson_blender pd;

    ///////////////////////////
    //ptr
    //vector<poisson_blender*> pds(mix_number, &pd);
    //vector<vector<string> *>bgs(mix_number, &bg);
    //vector<vector<vector<vector<string>>> *>obj_mask_labels(mix_number, &obj_mask_label);
    //vector<ofstream *>mix_msg_files(mix_number, &mix_msg_file);
    //vector<string*> save_image_paths(mix_number, &save_image_path);
    //vector<string*> save_xml_paths(mix_number, &save_xml_path);

    ///////////////////////
    //g_pd = pd;
    //g_bg = bg;
    //g_obj_mask_label = obj_mask_label;
    //g_mix_msg_file = mix_msg_file;
    //g_save_image_path = save_image_path;
    //g_save_xml_path = save_xml_path;
    ///////////////////////

    ///////////////////////////


    int begin_index = begin_line - 1;
 
    std::thread *p_thread = NULL;
    cv::Mat input_bg[3];
    vector<cv::Mat> input_obj_mask_label[3];
    Mat mix_img[3];
    vector<Rect> bd_set[3];
    vector<bool> is_blended[3];
    int flag_return[3] = { 1,1 ,1};
    bool is_save[3];

    int pre_kkk = 2;
    int kkk = 0;
    int next_kkk = 1;

    string save_image_path = save_path + "\\JPEGImages";
    string save_xml_path = save_path + "\\Annotations";

    //first
    if (begin_index < mix_number && begin_index < end_line)
    {
        //load img 
        bool is_continue = false;
        if (!load_img(bg[begin_index], obj_mask_label[begin_index][0], input_bg[kkk], input_obj_mask_label[kkk]))
        {
            cout << "line " << begin_index + 1 << "load fail." << endl;
            flag_return[kkk] = 0;
            is_continue = true;
        }
        //cal now
        if (!is_continue)
        {
            flag_return[kkk] = -1;
            p_thread = new std::thread(&poisson_blender::blend_thread,
                &input_bg[kkk], &input_obj_mask_label[kkk], &mix_img[kkk], &bd_set[kkk], &is_blended[kkk], &pd, &flag_return[kkk]);
            //poisson_blender::blend_thread(&input_bg[kkk], &input_obj_mask_label[kkk], &mix_img[kkk], &bd_set[kkk], &is_blended[kkk], &pd, &flag_return[kkk]);
        }
        //next
        {
            int ttt = kkk;
            kkk = next_kkk;
            next_kkk = pre_kkk;
            pre_kkk = ttt;
        }
    }

    //thread
    int i = begin_index + 1;
    for (; i < mix_number && i < end_line; ++i)
    {
        //clock_t begin, end;
        //begin = clock();
        //load img 
        bool is_continue = false;
        if (!load_img(bg[i], obj_mask_label[i][0], input_bg[kkk], input_obj_mask_label[kkk]))
        {
            cout << "line " << i + 1 << "load fail." << endl;
            flag_return[kkk] = 0;
            is_continue = true;
        }
        //wait pre cal
        {
            p_thread->join();
            delete p_thread;
            p_thread = NULL;
        }
        //cal now
        if (!is_continue)
        {
            flag_return[kkk] = -1;
            p_thread = new std::thread(&poisson_blender::blend_thread,
                &input_bg[kkk], &input_obj_mask_label[kkk], &mix_img[kkk], &bd_set[kkk], &is_blended[kkk], &pd, &flag_return[kkk]);
            //poisson_blender::blend_thread(&input_bg[kkk], &input_obj_mask_label[kkk], &mix_img[kkk], &bd_set[kkk], &is_blended[kkk], &pd, &flag_return[kkk]);
        }
        //save pre
        {
            string xml_name;
            xml_name = "fusion";
            xml_name += "_" + getToday();
            xml_name += "_" + to_string(i+index_param);
            is_save[pre_kkk] = pd.save(mix_img[pre_kkk], xml_name, save_image_path, save_xml_path, bd_set[pre_kkk], obj_mask_label[i-1][1]);

            string out_msg;
            out_msg = to_string(i) + "    ";
            if (flag_return[pre_kkk] == 1) { out_msg += "blend = succeed    "; }
            else
            {
                out_msg += "blend = fail in the index of ";
                out_msg += to_string(expand::find(is_blended[pre_kkk], false));
                out_msg += "    ";
            }
            if (is_save[pre_kkk]) { out_msg += "save = succeed    "; }
            else { out_msg += "save = fail    "; }

            cout << out_msg;

            //end = clock();
            //std::cout << "    time = " << double(end - begin) / CLOCKS_PER_SEC * 1000 << "ms    ";
            
            cout<< endl;
            mix_msg_file << out_msg << endl;
        }
        //next
        {
            int ttt = kkk;
            kkk = next_kkk;
            next_kkk = pre_kkk;
            pre_kkk = ttt;
        }

        //th_fun(&pd, i, &bg, &obj_mask_label, &mix_msg_file, &(save_path+"\\image"), &(save_path + "\\xml"));
    }

    //last
    //wait pre cal
    {
        p_thread->join();
        delete p_thread;
        p_thread = NULL;
    }
    //save pre
    {
        string xml_name;
        xml_name = "fusion";
        xml_name += "_" + getToday();
        xml_name += "_" + to_string(i+ index_param);
        is_save[pre_kkk] = pd.save(mix_img[pre_kkk], xml_name, save_image_path, save_xml_path, bd_set[pre_kkk], obj_mask_label[i - 1][1]);

        string out_msg;
        out_msg = to_string(i) + "    ";
        if (flag_return[pre_kkk] == 1) { out_msg += "blend = succeed    "; }
        else
        {
            out_msg += "blend = fail in the index of ";
            out_msg += to_string(expand::find(is_blended[pre_kkk], false));
            out_msg += "    ";
        }
        if (is_save[pre_kkk]) { out_msg += "save = succeed    "; }
        else { out_msg += "save = fail    "; }

        cout << out_msg << endl;
        mix_msg_file << out_msg << endl;
    }

    mix_msg_file.close();
    mix_msg_file.clear();

    //cout << "finish" << endl;

    //getchar();
    return 0;
}

