################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../src/allsat/utils/Options.cc \
../src/allsat/utils/System.cc 

CC_DEPS += \
./src/allsat/utils/Options.d \
./src/allsat/utils/System.d 

OBJS += \
./src/allsat/utils/Options.o \
./src/allsat/utils/System.o 


# Each subdirectory must supply rules for building sources it contributes
src/allsat/utils/%.o: ../src/allsat/utils/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++0x -DNDEBUG -DNLOGPRECO -DNSTATSPRECO -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -I../src/ -I../src/allsat/allsat -I../src/allsat/ -I../src/stlsoft/include -I../src/fastformat/include -I../unittests/gtest -I../unittests/gtest/include -O3 -Wextra -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


