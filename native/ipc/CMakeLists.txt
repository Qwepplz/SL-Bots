cmake_minimum_required(VERSION 3.20)

project(sl_bots_ipc LANGUAGES CXX)

option(SL_BOTS_ENABLE_SOURCEMOD "Build SourceMod native registration" OFF)
set(SL_BOTS_SOURCEMOD_SDK_DIR "" CACHE PATH "SourceMod SDK checkout")

set(SL_BOTS_SOURCES ipc_extension.cpp)
if (SL_BOTS_ENABLE_SOURCEMOD)
    if (NOT SL_BOTS_SOURCEMOD_SDK_DIR)
        message(FATAL_ERROR "SL_BOTS_SOURCEMOD_SDK_DIR is required when SourceMod support is enabled")
    endif()
    if (NOT EXISTS "${SL_BOTS_SOURCEMOD_SDK_DIR}/public/smsdk_ext.cpp")
        message(FATAL_ERROR "SourceMod SDK is missing public/smsdk_ext.cpp")
    endif()
    list(APPEND SL_BOTS_SOURCES
        sourcemod_extension.cpp
        "${SL_BOTS_SOURCEMOD_SDK_DIR}/public/smsdk_ext.cpp"
    )
endif()

add_library(sl_bots_ipc SHARED ${SL_BOTS_SOURCES})
set_target_properties(sl_bots_ipc PROPERTIES
    OUTPUT_NAME "sl_bots_ipc.ext"
    SUFFIX ".dll"
    PREFIX ""
)
target_compile_features(sl_bots_ipc PRIVATE cxx_std_17)
target_compile_definitions(sl_bots_ipc PRIVATE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    SL_BOTS_IPC_BUILD=1
)
target_include_directories(sl_bots_ipc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if (SL_BOTS_ENABLE_SOURCEMOD)
    target_compile_definitions(sl_bots_ipc PRIVATE
        SL_BOTS_SOURCEMOD_BUILD=1
        _CRT_SECURE_NO_WARNINGS
    )
    target_include_directories(sl_bots_ipc PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/sourcemod
        ${SL_BOTS_SOURCEMOD_SDK_DIR}/public
        ${SL_BOTS_SOURCEMOD_SDK_DIR}/public/amtl
        ${SL_BOTS_SOURCEMOD_SDK_DIR}/public/amtl/amtl
        ${SL_BOTS_SOURCEMOD_SDK_DIR}/sourcepawn/include
    )
    if (MSVC)
        target_compile_options(sl_bots_ipc PRIVATE /wd4100 /wd4244)
    endif()
endif()

if (MSVC)
    target_link_libraries(sl_bots_ipc PRIVATE synchronization)
    target_compile_options(sl_bots_ipc PRIVATE /W4 /WX /EHsc)
else()
    target_compile_options(sl_bots_ipc PRIVATE -Wall -Wextra -Werror)
endif()
