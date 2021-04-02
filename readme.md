# cuFast: A GPU acceleration of FAST algorithm

## Environment
cuda-10.2

GPU compute ability 61(GeForce MX250)

`
!modify Makefile to adapt to your environment!
`
## Dirs

- global

GPU acceleration only with global memory
- shared

GPU acceleration with shared memory and shared memory
- shared_stream

A simple stream acceleration test based on shared (not stable due to no synchronization between memory copy! ONLY STREAM TEST!!!)

## Usage

- global   

```
./fast_global -d [YOUR_IMAGE_DIR] 
e.g.
./fast_global -d ../assets/
```
- shared 

```
./fast_shared -d [YOUR_IMAGE_DIR] 
e.g.
./fast_shared -d ../assets/
```
- shared_stream 

to visualise the stream in NVIDIA Visual Profiler, use

```
nvvp ./fast_shared -d [YOUR_IMAGE_DIR] 
e.g.
nvvp./fast_shared -d ../assets/
```

## Details
https://tianyma.github.io/2020/12/19/pdf-cufast.html
