add_library(libbase STATIC
        libbase/algorithms/priority_queue.cpp
        libbase/fast_random.cpp
        libbase/stats.cpp
)

target_include_directories(libbase PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (BUILD_TESTING)
    add_executable(libbase_tests
            libbase/algorithms/priority_queue_tests.cpp
            libbase/stats_tests.cpp
    )

    target_link_libraries(libbase_tests PRIVATE
            libbase
            GTest::gtest_main
    )

    add_test(NAME libbase_tests COMMAND libbase_tests)
endif ()