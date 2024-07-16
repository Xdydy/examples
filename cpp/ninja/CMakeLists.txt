cmake_minimum_required(VERSION 3.10)
project(MyProject)

# 使用 Ninja 编译
# set(CMAKE_GENERATOR "Ninja")

# 添加子目录
add_subdirectory(lib1)
add_subdirectory(lib2)

add_executable(main main.cpp)

# 链接静态库 lib1 和 lib2
target_link_libraries(main lib1 lib2)

# 将可执行文件安装到指定目录
# install(TARGETS main DESTINATION bin)