## Description

Builds an edge map using canny edge detection.

This is [a port of the VapourSynth plugin TEdgeMask](https://github.com/dubhater/vapoursynth-tedgemask).

### Requirements:

- AviSynth 2.60 / AviSynth+ 3.4 or later

- Microsoft VisualC++ Redistributable Package 2022 (can be downloaded from [here](https://github.com/abbodi1406/vcredist/releases))

### Usage:

```
vsTEdgeMask (clip, float "threshY", float "threshU", float "threshV", int "type", int "link", float "scale", int "y", int "u", int "v", int "opt")
```

### Parameters:

- clip\
    A clip to process. It must have planar format, 8..16 bit integer sample type, and subsampling ratios of at most 2.
    
- threshY, threshU, threshV\
    Sets the magnitude thresholds.\
    If over this value then a sample will be considered an edge, and the output pixel will be set to the maximum value allowed by the format. Otherwise the output pixel will be set to 0.\
    Set this to 0 to output a magnitude mask instead of a binary mask.\
    Default: threshY = threshU = threshV = 8.0.

- type\
    Sets the type of first order partial derivative approximation that is used.\
    1: 2 pixel.\
    2: 4 pixel.\
    3: Same as type = 1.\
    4: Same as type = 2.\
    5: 6 pixel (Sobel operator).\
    Default: 2.
    
- link\
    Specifies whether luma to chroma linking, no linking, or linking of every plane to every other plane is used.\
    0: No linking. The three edge masks are completely independent.\
    1: Luma to chroma linking. If a luma pixel is considered an edge, the corresponding chroma pixel is also marked as an edge.\
    2: Every plane to every other plane. If a pixel is considered an edge in any plane, the corresponding pixels in all the other planes are also marked as edges.\
    This parameter has no effect when clip has only one plane, when any plane's threshold is 0, or when some planes are not processed.\
    This parameter can only be 0 or 2 when clip is RGB.\
    Default: 2 when clip is RGB, otherwise 1.
    
- scale\
    If the output is a magnitude mask (threshold is 0), it is scaled by this value.\
    Note that in TEMmod this parameter had three different, undocumented default values for the different mask types, which made it difficult to use the parameter without reading the source code.\
    Default: 1.0.
    
- y, u, v\
    Planes to process.\
    1: Return garbage.\
    2: Copy plane.\
    3: Process plane. Always process planes when clip is RGB.\
    Default: y = u = v = 3.
    
- opt\
    Sets which cpu optimizations to use.\
    -1: Auto-detect.\
    0: Use C++ code.\
    1: Use SSE2 code.\
    Default: -1.
    
### Building:

- Windows\
    Use solution files.

- Linux
    ```
    Requirements:
        - Git
        - C++11 compiler
        - CMake >= 3.16
    ```
    ```
    git clone https://github.com/Asd-g/AviSynth-vsTEdgeMask && \
    cd AviSynth-vsTEdgeMask && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    sudo make install
    ```
