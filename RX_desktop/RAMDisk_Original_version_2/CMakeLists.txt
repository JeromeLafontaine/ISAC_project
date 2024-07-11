cmake_minimum_required(VERSION 3.5)

project(USRP_X310_TX_RX)

option(UHD_USE_STATIC_LIBS OFF)

if(MSVC)
	add_compile_options(/W4)
else()
	#add_compile_options(-Wall -Wextra -Wshadow -Wpedantic -Wconversion -Wsign-conversion -Werror -pipe -O3 -march=native)
	#add_compile_options(-Wall -Wextra -Wshadow -Wpedantic -Wconversion -Wsign-conversion)
	add_compile_options(-Wall -Wextra -Wshadow -Wpedantic -Wconversion -Wsign-conversion -std=c++11)
endif()


find_package(Boost 1.46 REQUIRED program_options system thread)
find_package(UHD "3.15.0")

add_executable(adv_rx_to_file adv_rx_to_file.cpp)
add_executable(tx_waveforms_radar tx_waveforms_radar.cpp)

target_link_libraries(tx_waveforms_radar ${Boost_LIBRARIES} ${UHD_LIBRARIES})
target_link_libraries(adv_rx_to_file ${Boost_LIBRARIES} ${UHD_LIBRARIES})

set(CMAKE_BUILD_TYPE Release)

