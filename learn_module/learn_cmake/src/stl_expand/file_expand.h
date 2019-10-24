#ifndef FILE_EXPAND_H
#define FILE_EXPAND_H

#include<string>
#include<vector>

namespace std
{
    namespace expand
    {
        //bool to_relative(const string &input_path, string &output_path);//TODO
        //bool to_complete(const string &input_path, string &output_path);//TODO
        //bool path_cd(const string &now_path, const string &command, string &changed_path);//TODO


        enum FILE_LIST_TYPE
        {
            SCAN_ONLY_FILE_FULL_NAME = 0,
            SCAN_ONLY_FILE_FRONT_NAME = 1,
            SCAN_RELATIVE_PATH_FILE_FULL_NAME = 2,
            SCAN_ROOT_PATH_FILE_FULL_NAME = 3
        };

        void system_complete(const string &input,string &output);
        void get_path_last_name(const string &path, string &name);
        bool is_absolute(const string &path);
        bool is_relative(const string &path);

        bool get_parent_absolute_path(const string &path, string &parent_path);
        void get_cwd(string &path);

        bool is_regular_file(const string &path_fname);
        bool is_directory(const string &path);


        //Can not scan net work file.
        bool scan_one_path_file(const string &path, const string &accepted_expand_name,
            vector<string> & file_name, int mode = SCAN_ONLY_FILE_FULL_NAME);

        //Can not scan net work file.
        bool scan_one_path_directory(const string &path, vector<string> & directory_name, int mode = SCAN_RELATIVE_PATH_FILE_FULL_NAME);

        //temp[0] = relative_dir, temp[1] = file_front_name, temp[2] = file_expand_name
        //Can not scan net work file.
        bool scan_file_list(const string &root_path, const string &accepted_expand_name,
            vector<string> & file_list, int mode, vector<vector<string>> &temp);

        //Can not scan net work file.
        bool scan_file_list(const string &root_path, const string &accepted_expand_name,
            vector<string> & file_list, int mode, vector<string> & relative_dir, vector<string> &file_front_name, vector<string> &file_expand_name);

        //file io
        //*check out a list from text file, it must same part size in one line while is_col_first;
        bool check_out_list(const string &path_fname, const string &pattern, vector<vector<string>> &list, bool is_col_first = true);
        //*force the creation of a directory, it will fail while have no permission or have same file 
        bool create_directory(const string &path);
        //bool save_list(const string &path_fname, const char pattern,const vector<vector<string>> &list, bool is_col_first = false);
        //bool save_list(const string &path_fname, const string &pattern, const vector<vector<string>> &list, bool is_col_first = false);



    }




}


#endif // 
