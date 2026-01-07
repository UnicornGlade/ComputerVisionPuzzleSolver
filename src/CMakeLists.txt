add_library(libapp STATIC
        algorithms/find_segments.cpp
        algorithms/non_maximum_suppression.cpp
        algorithms/simplify_segments.cpp
)
target_link_libraries(libapp PUBLIC libimages)

if (BUILD_TESTING)
    add_executable(libapp_tests
            algorithms/find_segments_test.cpp
            algorithms/non_maximum_suppression_test.cpp
            algorithms/simplify_segments_test.cpp
    )
    target_link_libraries(libapp_tests PRIVATE libapp GTest::gtest_main)
    add_test(NAME libapp_tests COMMAND libapp_tests)
endif ()

add_executable(app main.cpp)
target_link_libraries(app PRIVATE libapp)
