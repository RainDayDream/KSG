cmake_minimum_required(VERSION 3.16)
project(KSG LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# The distance kernels use AVX-512 (F, BW, VNNI) and FMA; -march=native on an AVX-512 machine enables all of them.
option(KSG_NATIVE "Compile with -march=native" ON)
find_package(OpenMP REQUIRED)

add_library(ksg_common INTERFACE)
target_include_directories(ksg_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(ksg_common INTERFACE -O3)
if (KSG_NATIVE)
    target_compile_options(ksg_common INTERFACE -march=native)
else()
    target_compile_options(ksg_common INTERFACE -mavx512f -mavx512bw -mavx512vnni -mfma)
endif()
target_link_libraries(ksg_common INTERFACE OpenMP::OpenMP_CXX)

function(ksg_exe target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE ksg_common ${ARGN})
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
endfunction()

# search
ksg_exe(ksg_search        src/ksg_search.cpp faiss openblas spdlog fmt)
# KSG (in-memory sparse graph): k-NN graph -> truncate + reverse edges + connectivity repair
ksg_exe(build_KSG         src/build_KSG.cpp faiss openblas spdlog fmt stdc++fs)
# SSG (SSD graph): search-based construction, in-degree boost, sharded build for billion scale
ksg_exe(build_ssg_search  tools/build_ssg_search.cpp)
ksg_exe(boost_indegree    tools/boost_indegree.cpp)
ksg_exe(shard_split       tools/shard_split.cpp)
ksg_exe(build_ssg_shard   tools/build_ssg_shard.cpp)
ksg_exe(boost_shard       tools/boost_shard.cpp)
ksg_exe(indegree_count    tools/indegree_count.cpp)
# SSD layout + in-memory sidecars
ksg_exe(pack_sectors      tools/pack_sectors.cpp)
ksg_exe(build_colocated   tools/build_colocated.cpp)
ksg_exe(build_sq4         tools/build_sq4.cpp)
ksg_exe(build_sqf         tools/build_sqf.cpp)
ksg_exe(build_entry       tools/build_entry.cpp)
ksg_exe(permute_ksg       tools/permute_ksg.cpp)
# low-memory variants
ksg_exe(build_ksg_compact tools/build_ksg_compact.cpp)
