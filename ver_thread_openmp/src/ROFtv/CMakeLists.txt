cmake_minimum_required(VERSION 3.1.0)

add_library(ROFtv ROFtv.cpp)

target_link_libraries(ROFtv ${image_process_dependence})
