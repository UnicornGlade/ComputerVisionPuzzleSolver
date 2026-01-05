add_library(libbase STATIC
        libbase/fast_random.cpp
)

target_include_directories(libbase PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
