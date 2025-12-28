cmake_minimum_required(VERSION 3.16)
project(scnn_torch_backend VERSION 1.0.0 LANGUAGES CXX)

# Do not set the default build type automatically, leave it to the user
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type specified, defaulting to 'Release'.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Include model generation logic
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ModelGeneration.cmake)

# Define the library target
add_library(scnn_torch_backend STATIC
  src/scnn_torch_backend.cpp)

# Add model generation dependency to the library
add_model_generation_dependency(scnn_torch_backend)

# Specify include directories for the target
target_include_directories(scnn_torch_backend
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})

# Link libraries
target_link_libraries(scnn_torch_backend
  PRIVATE
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS})

target_compile_features(scnn_torch_backend PUBLIC cxx_std_17)

# Optionally install the library
install(TARGETS scnn_torch_backend
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# Install header files
install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Export targets and dependencies
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

# Export include directories for downstream packages
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)

  # Include engine symbolic link setup
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestingSetup.cmake)

  ament_add_gtest(test_scnn_torch_backend
    test/test_scnn_torch_backend.cpp)

  # Add OpenCV includes for the test
  target_include_directories(test_scnn_torch_backend PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})

  # Link libraries
  target_link_libraries(test_scnn_torch_backend
    scnn_torch_backend
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS})

  # Make the test depend on the symbolic link
  add_testing_dependency(test_scnn_torch_backend)
endif()

ament_package()
