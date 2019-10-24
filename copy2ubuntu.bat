::rm
rmdir /s/q ubuntu

::root
mkdir .\ubuntu
copy .\README.md .\ubuntu 
copy .\TODO.txt .\ubuntu
copy .\test_capfps.py .\ubuntu

::learn_module
mkdir .\ubuntu\learn_module

::project learn_module\interview
mkdir .\ubuntu\learn_module\interview
copy .\learn_module\interview\*.py .\ubuntu\learn_module\interview
copy .\learn_module\interview\ans_card.jpg .\ubuntu\learn_module\interview

::project learn_module\learn_pycpp
mkdir .\ubuntu\learn_module\learn_pycpp
copy .\learn_module\learn_pycpp\*.py .\ubuntu\learn_module\learn_pycpp
copy .\learn_module\learn_pycpp\*.h .\ubuntu\learn_module\learn_pycpp
copy .\learn_module\learn_pycpp\*.cpp .\ubuntu\learn_module\learn_pycpp

::project learn_module\learn_tf
mkdir .\ubuntu\learn_module\learn_tf
copy .\learn_module\learn_tf\*.py .\ubuntu\learn_module\learn_tf

::project learn_module\REFERENCE
mkdir .\ubuntu\learn_module\REFERENCE
copy .\learn_module\REFERENCE\*.py .\ubuntu\learn_module\REFERENCE

::project pyBoost
mkdir .\ubuntu\pyBoost
copy .\pyBoost\*.py .\ubuntu\pyBoost
copy .\pyBoost\colorRing.png .\ubuntu\pyBoost

::project pyFusion
mkdir .\ubuntu\pyFusion
copy .\pyFusion\*.py .\ubuntu\pyFusion
copy .\pyFusion\*.ini .\ubuntu\pyFusion
copy .\pyFusion\obj_size.txt .\ubuntu\pyFusion

::test_module
mkdir .\ubuntu\test_module
xcopy .\test_module\*.py .\ubuntu\test_module

::project tf_module
mkdir .\ubuntu\tf_module
copy .\tf_module\*.py .\ubuntu\tf_module

::project tools
mkdir .\ubuntu\tools
copy .\tools\*.py .\ubuntu\tools

::copy to ubuntu
::xcopy /E/Y .\ubuntu X:\

pause

