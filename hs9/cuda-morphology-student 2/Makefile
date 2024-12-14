SRC_PATH = ./src
OBJ_PATH = ./obj
BIN_PATH = ./bin
INC_PATH = ./include
OUT_PATH = ./output_images

NVCC = /usr/local/cuda/bin/nvcc
NVCC_INC_FLAG = -I/usr/local/cuda/include -I$(INC_PATH)
NVCC_LIB_FLAG = -L/usr/local/cuda/lib64 -lcudart -lcuda
NVCC_FLAGS = -O3 --gpu-architecture=sm_75 -lineinfo --Werror all-warnings

CC = gcc
CC_INC_FLAG = $(NVCC_INC_FLAG)
CC_LIB_FLAG = $(NVCC_LIB_FLAG)
CC_FLAGS =

# -- Final product ----------
PRODUCT = morpho

# -- src and obj List ----------
FILE := $(shell ls ${SRC_PATH} | grep '\.cu')
SRC = $(addprefix ${SRC_PATH}/, $(FILE))
OBJ = $(addprefix ${OBJ_PATH}/, $(addsuffix .o, $(basename $(FILE))))

# -- Base rules ----------
$(OBJ_PATH)/%.o : $(SRC_PATH)/%.c
	$(NVCC) $(CC_FLAGS) $(CC_INC_FLAG) -c $< -o $@

# -- Cuda Base rules ----------
$(OBJ_PATH)/%.o : $(SRC_PATH)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INC_FLAG) -c $< -o $@

#-----Main rule ----------
$(BIN_PATH)/$(PRODUCT): $(OBJ)
	$(NVCC) -o $@ $^ $(CC_FLAGS) $(CC_INC_FLAG) $(CC_LIB_FLAG)

clean:
	rm -f $(OBJ)
	rm -f ${BIN_PATH}/${PRODUCT}
	rm -f ${OUT_PATH}/*
