::rm
rmdir /s/q ubuntu

::root
mkdir .\ubuntu
copy .\dependency.txt .\ubuntu
copy .\TODO.txt .\ubuntu
copy .\test_capfps.py .\ubuntu

::project Common
mkdir .\ubuntu\Common
copy .\Common\*.py .\ubuntu\Common
copy .\Common\colorRing.png .\ubuntu\Common

::project ImageTool
mkdir .\ubuntu\ImageTool
copy .\ImageTool\*.py .\ubuntu\ImageTool

::project interview
mkdir .\ubuntu\interview
copy .\interview\*.py .\ubuntu\interview
copy .\interview\ans_card.jpg .\ubuntu\interview

::project learn_tf
mkdir .\ubuntu\learn_tf
copy .\learn_tf\*.py .\ubuntu\learn_tf
copy .\learn_tf\ans_card.jpg .\ubuntu\learn_tf

::project pyBoost
mkdir .\ubuntu\pyBoost
copy .\pyBoost\*.py .\ubuntu\pyBoost
copy .\pyBoost\interview .\ubuntu\pyBoost

::project pyFusion
mkdir .\ubuntu\pyFusion
copy .\pyFusion\*.py .\ubuntu\pyFusion
copy .\pyFusion\*.ini .\ubuntu\pyFusion
copy .\pyFusion\obj_size.txt .\ubuntu\pyFusion

::REFERENCE
mkdir .\ubuntu\REFERENCE
xcopy .\REFERENCE .\ubuntu\REFERENCE

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
xcopy /E/Y .\ubuntu X:\

