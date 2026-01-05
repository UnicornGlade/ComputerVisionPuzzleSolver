add_library(libimages STATIC
        libimages/image.cpp
        libimages/image_io.cpp
        libimages/debug_io.cpp
)

target_include_directories(libimages PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(libimages PUBLIC libbase PRIVATE third_party_stb)
