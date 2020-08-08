cmake_minimum_required(VERSION 3.1.0)

#find_package(Qt5Widgets)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)  

if(OPENMP_FOUND)  
    message("OPENMP FOUND")  
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
endif()  

set(CMAKE_CXX_STANDARD 11)
set(image_process_dependence #Qt5::Widgets 
                             ${OpenCV_LIBS} 
                             ${CMAKE_THREAD_LIBS_INIT})

add_subdirectory(ROFtv)
add_subdirectory(app)


