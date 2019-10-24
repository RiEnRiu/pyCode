#ifndef CHRONO_EXPAND_H
#define CHRONO_EXPAND_H

#include<chrono>
#include<vector>
namespace std
{
    namespace expand
    {
        class chrono_time 
        {
        private:
            int ft_size = 60;
            vector<chrono::steady_clock::time_point> v_tp;
            vector<chrono::duration<double>> v_dura;
            int pre_ft_i;
            int ft_i;
            int next_ft_i;

            void fun_update();

        public:
            chrono_time(size_t _size = 60);

            ~chrono_time();

            double fps(bool update = 1);

            //ms
            double dura(bool update = 1);


        };

    }




}


#endif // 
