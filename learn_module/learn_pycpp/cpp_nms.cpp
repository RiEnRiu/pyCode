
#ifndef CPP_NMS_CPP
#define CPP_NMS_CPP

#define BOOST_PYTHON_STATIC_LIB
#include <vector>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "cpp_nms.h"

//#include<iostream>



namespace py = boost::python;
namespace np = py::numpy;

py::object cpp_nms(np::ndarray boxes, np::ndarray scores, py::object thresh)
{
    //py->cpp, boxes
    vector<vector<float>> cpp_boxes;
    for (size_t i = 0; i < py::len(boxes); ++i)
    {
        //std::cout<< "here!"<<std::endl;
        //std::cout<< py::type(boxes) <<std::endl;
        np::ndarray box = py::extract<np::ndarray>(boxes[i]);
        vector<float> t;
        for (size_t j = 0; j < py::len(box); ++j)
        {
            t.push_back(py::extract<float>(box[j]));
        }
        cpp_boxes.push_back(t);
    }
    //py->cpp, scores
    vector<float> cpp_scores;
    for (size_t i = 0; i < py::len(scores); ++i)
    {
        cpp_scores.push_back(py::extract<float>(scores[i]));
    }
    //py->cpp, thresh
    float cpp_thresh = py::extract<float>(thresh);

    //std::cout<< cpp_boxes.size()<<std::endl;
    //std::cout<< cpp_scores.size()<<std::endl;
    //std::cout<< cpp_thresh<<std::endl;

    //run
    auto cpp_keep = nms(cpp_boxes, cpp_scores,cpp_thresh);

    //cpp->py
    py::list keep;
    for(auto p=cpp_keep.begin();p!=cpp_keep.end();++p)
    {
        keep.append(*p);
    }
    return keep;
}


BOOST_PYTHON_MODULE(cpp_nms)
{
    using namespace boost::python;
    np::initialize();
    def("cpp_nms", cpp_nms);
}


#endif // !CPP_NMS_CPP


