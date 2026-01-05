add_library(libimages STATIC
        libimages/algorithms/downscale.cpp
        libimages/algorithms/gaussian_blur.cpp
        libimages/algorithms/grayscale.cpp
        libimages/algorithms/sobel_gradients.cpp
        libimages/debug_io.cpp
        libimages/image.cpp
        libimages/image_io.cpp
)

target_include_directories(libimages PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(libimages PUBLIC libbase PRIVATE third_party_stb)
