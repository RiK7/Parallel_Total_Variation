cmake_minimum_required(VERSION 3.1.0)

#find_package(Qt5Widgets)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(image_process_dependence #Qt5::Widgets 
                             ${OpenCV_LIBS} 
                             ${CMAKE_THREAD_LIBS_INIT})

add_subdirectory(ROFtv)
add_subdirectory(app)


