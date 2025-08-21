cmake_minimum_required(VERSION 3.10)
project(ControlCamera)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)               # Enable moc autogeneration
set(CMAKE_AUTOUIC ON)               # If you use Qt Designer .ui files
set(CMAKE_AUTORCC ON)               # If you use Qt resource files

find_package(Qt6 COMPONENTS Core Widgets Gui REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(ControlCamera
    main.cpp
    mainwindow.cpp
    ControlCamera.cpp
    mainwindow.h
    ControlCamera.h
)

target_link_libraries(ControlCamera PRIVATE Qt6::Core Qt6::Widgets Qt6::Gui ${OpenCV_LIBS})
