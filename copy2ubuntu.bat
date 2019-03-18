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

::project handDetect
mkdir .\ubuntu\handDetect
copy .\handDetect\*.py .\ubuntu\handDetect

::project ImageTool
mkdir .\ubuntu\ImageTool
copy .\ImageTool\*.py .\ubuntu\ImageTool

::project imTool
mkdir .\ubuntu\imTool
copy .\imTool\*.py .\ubuntu\imTool

::project interview
mkdir .\ubuntu\interview
copy .\interview\*.py .\ubuntu\interview
copy .\interview\ans_card.jpg .\ubuntu\interview

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

::project tracker
mkdir .\ubuntu\tracker
copy .\tracker\*.py .\ubuntu\tracker

::project tracker
mkdir .\ubuntu\tracker
copy .\tracker\*.py .\ubuntu\tracker

::project videoTool
mkdir .\ubuntu\videoTool
copy .\videoTool\*.py .\ubuntu\videoTool

::project vocTool
mkdir .\ubuntu\vocTool
copy .\vocTool\*.py .\ubuntu\vocTool

::copy to ubuntu
xcopy /E/Y .\ubuntu X:\

