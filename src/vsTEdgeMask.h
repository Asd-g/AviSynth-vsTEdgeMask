#pragma once

#include "avisynth.h"

enum MaskTypes
{
    TwoPixel = 1,
    FourPixel = 2,
    SixPixel = 5
};

template<typename PixelType, MaskTypes type, bool binarize>
void detect_edges_sse2(const uint8_t* srcp8, uint8_t* dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max);

class vsTEdgeMask : public GenericVideoFilter
{
    int64_t _threshold[3];
    int _link;
    double _scale;
    int process[3];
    bool has_at_least_v8;

    void (*detect_edges[3])(const uint8_t* srcp8, uint8_t* dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max);
    void (*link_planes)(uint8_t* dstp18, uint8_t* dstp28, uint8_t* dstp38, int stride1, int stride2, int width, int height, int pixel_max);

public:
    vsTEdgeMask(PClip _child, double threshY, double threshU, double threshV, int type, int link, float scale, int y, int u, int v, int opt, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
    int __stdcall SetCacheHints(int cachehints, int frame_range)
    {
        return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
    }
};
