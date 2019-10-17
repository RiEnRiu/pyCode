
#ifndef CPP_NMS_H
#define CPP_NMS_H

#include <vector>
#include <algorithm>
#include <numeric>
using namespace std;

#ifndef MAX
#define MAX(a,b) (a)>(b)?(a):(b)
#endif

#ifndef MIN
#define MIN(a,b) (a)<(b)?(a):(b)
#endif

template <typename T>
vector<size_t> argsort(const vector<T> &v) 
{
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    return idx;
}

vector<size_t> nms(const vector<vector<float>> &boxes, const vector<float> &scores, float thresh)
{
    int nboxes = boxes.size();

    vector<float> areas(nboxes);
    for(int i=0;i<nboxes;++i)
    {
        auto box = boxes[i];
        areas[i] = (box[2]-box[0])*(box[3]-box[1]);
    }
    auto order = argsort(scores);
    auto tmp_order = order;

    vector<size_t> keep;

    while(order.size()>0)
    {
        size_t i = *(order.end()-1);
        keep.push_back(i);
        auto keep_box = boxes[i];
        tmp_order.clear();
        for(auto pi = order.begin();pi!=order.end()-1;++pi)
        {
            size_t _i = *pi;
            auto box = boxes[_i];
            float xx1 = MAX(keep_box[0], box[0]);
            float yy1 = MAX(keep_box[1], box[1]);
            float xx2 = MIN(keep_box[2], box[2]);
            float yy2 = MIN(keep_box[3], box[3]);
            float w = MAX(0, xx2-xx1);
            float h = MAX(0, yy2-yy1);
            float inter = w*h;
            float ovr = inter/(areas[i]+areas[_i]-inter);
            if(ovr<=thresh)
            {
                tmp_order.push_back(_i);
            }
        }
        tmp_order.swap(order);
    }
    return keep;
}

#endif // !CPP_NMS_H


