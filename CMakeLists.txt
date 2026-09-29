cmake_minimum_required(VERSION 3.5)
project(torch_ray)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

add_executable(torch_ray
    app/main.cpp
    src/models.cpp
    src/renderer.cpp
    src/utils.cpp
)

# add_executable(torch_ray
#     test_files/image_preview.cpp
#     src/models.cpp
#     src/renderer.cpp
#     src/utils.cpp
# )


target_include_directories(torch_ray PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(torch_ray
    ${TORCH_LIBRARIES}
)
