::rm
rmdir /s/q ubuntu

::root
mkdir .\ubuntu
copy .\README.md .\ubuntu 
copy .\TODO.txt .\ubuntu

::file
mkdir .\ubuntu\file
copy .\file\* .\ubuntu\file

::project REFERENCE
mkdir .\ubuntu\REFERENCE
copy .\REFERENCE\* .\ubuntu\REFERENCE

::project pyBoost
mkdir .\ubuntu\pyBoost
copy .\pyBoost\*.py .\ubuntu\pyBoost

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

