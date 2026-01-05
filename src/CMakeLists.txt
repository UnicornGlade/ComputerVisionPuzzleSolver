add_executable(sobel_app main.cpp disjoint_set.cpp)
target_link_libraries(sobel_app PRIVATE libimages)
