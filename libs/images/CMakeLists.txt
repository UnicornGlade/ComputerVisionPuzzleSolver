add_library(libimages STATIC
        libimages/algorithms/downscale.cpp
        libimages/algorithms/gaussian_blur.cpp
        libimages/algorithms/grayscale.cpp
        libimages/algorithms/sobel_gradients.cpp
        libimages/debug_io.cpp
        libimages/draw.cpp
        libimages/image.cpp
        libimages/image_io.cpp
)

target_include_directories(libimages PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(libimages PUBLIC libbase PRIVATE third_party_stb)

if (BUILD_TESTING)
    add_executable(libimages_tests
            libimages/draw_tests.cpp
    )
    target_link_libraries(libimages_tests PRIVATE libimages GTest::gtest_main)
    add_test(NAME libimages_tests COMMAND libimages_tests)
endif ()
