# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
  "CMakeFiles/ControlCamera_autogen.dir/AutogenUsed.txt"
  "CMakeFiles/ControlCamera_autogen.dir/ParseCache.txt"
  "ControlCamera_autogen"
  )
endif()
