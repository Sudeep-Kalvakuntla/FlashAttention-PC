NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_89 -lineinfo

attention: src/attention.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f attention attention.exe

.PHONY: clean
