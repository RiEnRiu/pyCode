
#include"chrono_expand.h"

namespace std
{
    namespace expand
    {
        void chrono_time::fun_update()
        {
            v_tp[ft_i] = chrono::steady_clock::now();
            v_dura[ft_i] = chrono::duration_cast<chrono::duration<double>>(v_tp[ft_i] - v_tp[pre_ft_i]);
            pre_ft_i = (pre_ft_i + 1) % ft_size;
            ft_i = (ft_i + 1) % ft_size;
            next_ft_i = (next_ft_i + 1) % ft_size;
        }

        chrono_time::chrono_time(size_t _size)
        {
            ft_size = _size;
            v_tp.resize(_size);
            v_dura.resize(_size);

            pre_ft_i = _size - 1;
            ft_i = 0;
            next_ft_i = 1;

            v_tp[0] = chrono::steady_clock::now();
            v_dura[0] = chrono::duration_cast<chrono::duration<double>>(v_tp[0] - v_tp[0]);
            for (int i = 1; i < _size; ++i) { v_tp[i] = v_tp[0]; v_dura[i] = v_dura[0]; }
        }

        chrono_time::~chrono_time() {}

        double chrono_time::fps(bool update)
        {
            if (update) { fun_update(); }
            return ft_size / chrono::duration_cast<chrono::duration<double>>(v_tp[pre_ft_i] - v_tp[ft_i]).count();
        }

        double chrono_time::dura(bool update)
        {
            if (update) { fun_update(); }
            return v_dura[pre_ft_i].count()*1000;
        }


    }
}
