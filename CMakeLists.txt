cmake_minimum_required(VERSION 2.8)
project(streetSigns)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}
  src/main.cpp
  src/StreetSignsIdentifier.cpp
  src/StreetSign.cpp
  src/StreetSign_Warning.cpp
  src/StreetSign_NoParking.cpp
  src/StreetSign_Speed.cpp
  src/utils.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
