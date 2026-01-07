add_library(libapp STATIC
        algorithms/background_masking.cpp
        algorithms/find_mask_borders.cpp
        algorithms/utils.cpp
)
target_link_libraries(libapp PUBLIC libimages)

if (BUILD_TESTING)
    add_executable(libapp_tests
            algorithms/background_masking_tests.cpp
            algorithms/find_mask_borders_tests.cpp
            algorithms/utils_tests.cpp
    )
    target_link_libraries(libapp_tests PRIVATE libapp GTest::gtest_main)
    add_test(NAME libapp_tests COMMAND libapp_tests)
endif ()

add_executable(app main.cpp)
target_link_libraries(app PRIVATE libapp)
