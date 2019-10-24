#ifndef VECTOR_EXPAND_HPP
#define VECTOR_EXPAND_HPP

////////////////////////////

#include<vector>
#include<random>
#include<iostream>
#include<iomanip>

#ifndef VECTOR_NPOS
#define VECTOR_NPOS size_t(-1)
#endif 





namespace std
{
    namespace expand
    {



        template <typename T>
        inline void clearVector(vector<T> &src)
        {
            vector<T>(src).swap(src);
            return;
        }


        template <typename T>
        void transpose(vector<vector<T>> vv_input, vector<vector<T>> &vv_output, void(*copy_func)(const T &, T&) = NULL)
        {
            if (copy_func == NULL)
            {
                size_t input_size = vv_input.size();
                size_t max_size = 0;
                for (int i = 0; i < input_size; ++i)
                {
                    size_t one_size = vv_input[i].size();
                    if (one_size > max_size) { max_size = one_size; }
                }
                vv_output.resize(max_size);
                for (int j = 0; j < max_size; ++j) { vv_output[j].resize(input_size); }
                for (int i = 0; i < vv_input.size(); ++i)
                {
                    size_t one_size = vv_input[i].size();
                    for (int j = 0; j < one_size; ++j)
                    {
                        vv_output[j][i] = vv_input[i][j];
                    }
                }
                return;
            }
            else
            {
                size_t input_size = vv_input.size();
                size_t max_size = 0;
                for (int i = 0; i < input_size; ++i)
                {
                    size_t one_size = vv_input[i].size();
                    if (one_size > max_size) { max_size = one_size; }
                }
                vv_output.resize(max_size);
                for (int j = 0; j < max_size; ++j) { vv_output[j].resize(input_size); }
                for (int i = 0; i < vv_input.size(); ++i)
                {
                    size_t one_size = vv_input[i].size();
                    for (int j = 0; j < one_size; ++j)
                    {
                        copy_func(vv_input[i][j], vv_output[j][i]);
                    }
                }
                return;
            }
        }


        template <typename T1, typename T2>
        void convert(const vector<T1> &v_input, vector<T2> &v_output, void(*deep_copy_func)(const T1 &, T2 &) = NULL)
        {
            if (deep_copy_func == NULL)
            {
                v_output.resize(v_input.size());
                for (int i = 0; i < v_input.size(); ++i)
                {
                    v_output[i] = v_input[i];
                }
            }
            else
            {
                v_output.resize(v_input.size());
                for (int i = 0; i < v_input.size(); ++i)
                {
                    deep_copy_func(v_input, v_output);
                }
            }
            return;
        }

        template <typename T>
        size_t choiceFrom(const vector<T> &v_T, T &choice_value, void(*copy_func)(const T &, T &) = NULL)
        {
            if (v_T.size() == 0) return VECTOR_NPOS;
            static std::default_random_engine rand_e(time(0));
            std::uniform_int_distribution<size_t> unif_dist(0, v_T.size() - 1);
            size_t index = unif_dist(rand_e);
            if (copy_func == NULL) choice_value = v_T[index];
            else copy_func(v_T[index], choice_value);
            return index;
        }

        template <typename T>
        size_t find(const vector<T> &v_T, const T &_value, bool(*compare_func)(const T &, const T &) = NULL)
        {
            if (compare_func == NULL)
            {
                for (int i = 0; i < v_T.size(); ++i)
                    if (_value == v_T[i]) { return i; }
            }
            else
            {
                for (int i = 0; i < v_T.size(); ++i)
                    if (compare_func(_value, v_T[i])) { return i; }
            }
            return VECTOR_NPOS;
        }

        template <typename T>
        inline void print_col(const vector<T> &v_T, int _w = 6)
        {
            for (int i = 0; i < v_T.size(); ++i)
            {
                cout << setw(_w) << v_T[i] << endl;
            }
            return;
        }

        template <typename T>
        inline void print_row(const vector<T> &v_T, int space_number, int align_right_width = 0)
        {
            for (int i = 0; i < v_T.size(); ++i)
            {
                cout << setw(align_right_width) << v_T[i] << setw(space_number) << " ";
            }
            cout << endl;
            return;
        }

        template <typename T>
        inline void print_list(const vector<vector<T>> &v_v_T, int space_number, int align_right_width = 0)
        {
            for (int i = 0; i < v_v_T.size(); ++i)
            {
                for (int j = 0; j < v_v_T[i].size(); ++j)
                {
                    cout << setw(align_right_width) << v_v_T[i][j] << setw(space_number) << " ";
                }
                cout << endl;
            }
            return;
        }




    }

}





#endif // 
