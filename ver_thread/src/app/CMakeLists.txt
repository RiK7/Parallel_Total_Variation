cmake_minimum_required(VERSION 3.1.0)
project(app)


include_directories(${CMAKE_SOURCE_DIR})


add_executable(ROF_tv ROF_tv.cpp)

target_link_libraries(ROF_tv ROFtv ${image_process_dependence})
