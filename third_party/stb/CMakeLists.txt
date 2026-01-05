# Always provide a target named third_party_stb.
# - If USE_SYSTEM_IMAGE_LIBS=OFF: build stb as a static library (compiled once).
# - If USE_SYSTEM_IMAGE_LIBS=ON: provide an interface target to keep link graph stable.

if (USE_SYSTEM_IMAGE_LIBS)
    add_library(third_party_stb INTERFACE)
    target_include_directories(third_party_stb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
else ()
    add_library(third_party_stb STATIC stb_impl.cpp)
    target_include_directories(third_party_stb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif ()
