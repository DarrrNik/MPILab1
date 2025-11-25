cmake_minimum_required(VERSION 3.20)
project(MPILab1_task2 LANGUAGES CXX)

find_package(MPI REQUIRED)

# Версия стандарта C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Папка, куда будут складываться все .exe
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/executables")
file(MAKE_DIRECTORY "${CMAKE_OUTPUT_PATH}")

# Создание исполняемых файлов
add_executable(${PROJECT_NAME}-row   mpilab.cpp row.cpp)
add_executable(${PROJECT_NAME}-col   mpilab.cpp col.cpp)
add_executable(${PROJECT_NAME}-block mpilab.cpp block.cpp)

# Линкуем MPI корректно
target_link_libraries(${PROJECT_NAME}-row   PRIVATE MPI::MPI_CXX)
target_link_libraries(${PROJECT_NAME}-col   PRIVATE MPI::MPI_CXX)
target_link_libraries(${PROJECT_NAME}-block PRIVATE MPI::MPI_CXX)

# Включение include MPI
include_directories(${MPI_CXX_INCLUDE_PATH})

# Опционально: вывод компиляционных команд (удобно для языковых серверов)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Опционально: включить строгие предупреждения (как у Clang/GCC/MSVC)
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

