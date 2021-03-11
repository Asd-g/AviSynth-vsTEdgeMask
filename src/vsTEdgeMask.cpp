#include <algorithm>
#include <cmath>

#include "vsTEdgeMask.h"

enum LinkModes
{
    LinkNothing,
    LinkChromaToLuma,
    LinkEverything
};

template<typename PixelType, MaskTypes type, bool binarize>
static void detect_edges_scalar(const uint8_t* srcp8, uint8_t* dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max)
{
    const PixelType* srcp = reinterpret_cast<const PixelType*>(srcp8);
    PixelType* dstp = reinterpret_cast<PixelType*>(dstp8);
    stride /= sizeof(PixelType);
    dst_stride /= sizeof(PixelType);
    width /= sizeof(PixelType);

    typedef typename std::conditional<sizeof(PixelType) == 1, int32_t, int64_t>::type int32_or_64;

    int32_or_64 threshold = (int32_or_64)threshold64;

    // Number of pixels to skip at the edges of the image.
    const int skip = type == FourPixel ? 2 : 1;

    for (int y = 0; y < skip; ++y)
    {
        memset(dstp, 0, width * sizeof(PixelType));

        srcp += stride;
        dstp += dst_stride;
    }

    for (int y = skip; y < height - skip; ++y)
    {
        memset(dstp, 0, skip * sizeof(PixelType));

        for (int x = skip; x < width - skip; ++x)
        {
            int32_or_64 gx, gy;
            float divisor;

            int top = srcp[x - stride];
            int left = srcp[x - 1];
            int right = srcp[x + 1];
            int bottom = srcp[x + stride];

            if (type == TwoPixel)
            {
                gx = static_cast<int64_t>(right) - left;
                gy = static_cast<int64_t>(top) - bottom;
                divisor = 0.25f;
            }
            else if (type == FourPixel)
            {
                int top2 = srcp[x - 2 * stride];
                int left2 = srcp[x - 2];
                int right2 = srcp[x + 2];
                int bottom2 = srcp[x + 2 * stride];

                gx = 12 * (static_cast<int64_t>(left2) - right2) + 74 * (static_cast<int64_t>(right) - left);
                gy = 12 * (static_cast<int64_t>(bottom2) - top2) + 74 * (static_cast<int64_t>(top) - bottom);
                divisor = 0.0001f;
            }
            else if (type == SixPixel)
            {
                int top_left = srcp[x - stride - 1];
                int top_right = srcp[x - stride + 1];
                int bottom_left = srcp[x + stride - 1];
                int bottom_right = srcp[x + stride + 1];

                gx = top_right + static_cast<int64_t>(2) * right + bottom_right - top_left - static_cast<int64_t>(2) * left - bottom_left;
                gy = bottom_left + static_cast<int64_t>(2) * bottom + bottom_right - top_left - static_cast<int64_t>(2) * top - top_right;
                divisor = 1.0f;
            }

            int32_or_64 sum_squares = gx * gx + gy * gy;

            if (binarize)
                dstp[x] = (sum_squares > threshold) ? pixel_max : 0;
            else
                dstp[x] = std::min((int)(std::sqrt(sum_squares * divisor) * scale + 0.5f), pixel_max);
        }

        memset(dstp + width - skip, 0, skip * sizeof(PixelType));

        srcp += stride;
        dstp += dst_stride;
    }

    for (int y = height - skip; y < height; ++y)
    {
        memset(dstp, 0, width * sizeof(PixelType));

        srcp += stride;
        dstp += dst_stride;
    }
}

template<typename PixelType, LinkModes link>
static void link_planes_444_scalar(uint8_t* dstp18, uint8_t* dstp28, uint8_t* dstp38, int stride1, int stride2, int width, int height, int pixel_max)
{
    (void)stride2;
    (void)pixel_max;

    PixelType* dstp1 = reinterpret_cast<PixelType*>(dstp18);
    PixelType* dstp2 = reinterpret_cast<PixelType*>(dstp28);
    PixelType* dstp3 = reinterpret_cast<PixelType*>(dstp38);
    stride1 /= sizeof(PixelType);
    width /= sizeof(PixelType);


    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            PixelType val = dstp1[x];

            if (link == LinkEverything)
            {
                val |= dstp2[x] | dstp3[x];

                if (val)
                    dstp1[x] = val;
            }

            if (val)
                dstp2[x] = dstp3[x] = val;
        }

        dstp1 += stride1;
        dstp2 += stride1;
        dstp3 += stride1;
    }
}

template<typename PixelType, LinkModes link>
static void link_planes_422_scalar(uint8_t* dstp18, uint8_t* dstp28, uint8_t* dstp38, int stride1, int stride2, int width, int height, int pixel_max)
{
    (void)pixel_max;

    PixelType* dstp1 = reinterpret_cast<PixelType*>(dstp18);
    PixelType* dstp2 = reinterpret_cast<PixelType*>(dstp28);
    PixelType* dstp3 = reinterpret_cast<PixelType*>(dstp38);
    stride1 /= sizeof(PixelType);
    stride2 /= sizeof(PixelType);
    width /= sizeof(PixelType);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; x += 2)
        {
            PixelType val = dstp1[x] & dstp1[x + 1];

            if (link == LinkEverything)
            {
                val |= dstp2[x >> 1] | dstp3[x >> 1];

                if (val)
                    dstp1[x] = dstp1[x + 1] = val;
            }

            if (val)
                dstp2[x >> 1] = dstp3[x >> 1] = val;
        }

        dstp1 += stride1;
        dstp2 += stride2;
        dstp3 += stride2;
    }
}

template<typename PixelType, LinkModes link>
static void link_planes_440_scalar(uint8_t* dstp18, uint8_t* dstp28, uint8_t* dstp38, int stride1, int stride2, int width, int height, int pixel_max)
{
    (void)stride2;
    (void)pixel_max;

    PixelType* dstp1 = reinterpret_cast<PixelType*>(dstp18);
    PixelType* dstp2 = reinterpret_cast<PixelType*>(dstp28);
    PixelType* dstp3 = reinterpret_cast<PixelType*>(dstp38);
    stride1 /= sizeof(PixelType);
    width /= sizeof(PixelType);

    for (int y = 0; y < height; y += 2)
    {
        for (int x = 0; x < width; ++x)
        {
            PixelType val = dstp1[x] & dstp1[x + stride1];

            if (link == LinkEverything)
            {
                val |= dstp2[x] | dstp3[x];

                if (val)
                    dstp1[x] = dstp1[x + stride1] = val;
            }

            if (val)
                dstp2[x] = dstp3[x] = val;
        }

        dstp1 += stride1 * static_cast<int64_t>(2);
        dstp2 += stride1;
        dstp3 += stride1;
    }
}

template<typename PixelType, LinkModes link>
static void link_planes_420_scalar(uint8_t* dstp18, uint8_t* dstp28, uint8_t* dstp38, int stride1, int stride2, int width, int height, int pixel_max)
{
    PixelType* dstp1 = reinterpret_cast<PixelType*>(dstp18);
    PixelType* dstp2 = reinterpret_cast<PixelType*>(dstp28);
    PixelType* dstp3 = reinterpret_cast<PixelType*>(dstp38);
    stride1 /= sizeof(PixelType);
    stride2 /= sizeof(PixelType);
    width /= sizeof(PixelType);

    for (int y = 0; y < height; y += 2)
    {
        for (int x = 0; x < width; x += 2)
        {
            int sum = 0;

            if (dstp1[x])
                ++sum;
            if (dstp1[x + 1])
                ++sum;
            if (dstp1[x + stride1])
                ++sum;
            if (dstp1[x + stride1 + 1])
                ++sum;

            if (link == LinkEverything)
            {
                if (dstp2[x >> 1])
                    sum += 2;
                if (dstp3[x >> 1])
                    sum += 2;

                if (sum > 1)
                    dstp1[x] = dstp1[x + 1] = dstp1[x + stride1] = dstp1[x + stride1 + 1] = pixel_max;
            }

            if (sum > 1)
                dstp2[x >> 1] = dstp3[x >> 1] = pixel_max;
        }

        dstp1 += stride1 * static_cast<int64_t>(2);
        dstp2 += stride2;
        dstp3 += stride2;
    }
}

vsTEdgeMask::vsTEdgeMask(PClip _child, double threshY, double threshU, double threshV, int type, int link, float scale, int y, int u, int v, int opt, IScriptEnvironment* env)
    : GenericVideoFilter(_child), _link(link), _scale(scale)
{
    const int bits = vi.BitsPerComponent();

    const int subsw = (vi.IsY() || vi.IsRGB()) ? 0 : vi.GetPlaneWidthSubsampling(PLANAR_U);
    const int subsh = (vi.IsY() || vi.IsRGB()) ? 0 : vi.GetPlaneHeightSubsampling(PLANAR_U);

    if (!vi.IsPlanar())
        env->ThrowError("vsTEdgeMask: clip must be in planar format.");
    if (bits > 16 || subsw > 1 || subsh > 1)
        env->ThrowError("vsTEdgeMask: clip must have constant format, 8..16 bit integer samples, and subsampling ratios of at most 2.");
    if (threshY < 0)
        env->ThrowError("vsTEdgeMask: threshY must not be negative.");
    if (threshU < 0)
        env->ThrowError("vsTEdgeMask: threshU must not be negative.");
    if (threshV < 0)
        env->ThrowError("vsTEdgeMask: threshV must not be negative.");
    if (type < 1 || type > 5)
        env->ThrowError("vsTEdgeMask: type must be between 1 and 5 (inclusive).");

    // Types 3 and 4 from TEMmod are not implemented.
    // They are more or less types 1 and 2 but with integer SSE2 code.
    if (type == 3)
        type = TwoPixel;
    if (type == 4)
        type = FourPixel;

    if (_link < 0 || _link > 2)
        env->ThrowError("vsTEdgeMask: link must be 0, 1, or 2.");
    if (_link == LinkChromaToLuma && vi.IsRGB()) 
        env->ThrowError("vsTEdgeMask: link must be 0 or 2 when clip is RGB.");
    if (_scale < 0.0f)
        env->ThrowError("vsTEdgeMask: scale must not be negative.");
    if (opt > 1 || opt < -1)
        env->ThrowError("vsTEdgeMask: opt msut be between -1..1.");
    if (opt == 1 && !(env->GetCPUFlags() & CPUF_SSE2))
        env->ThrowError("vsTEdgeMask: opt=1 requires SSE2.");
    if (y < 1 || y > 3)
        env->ThrowError("vsTEdgeMask: y must be between 1..3.");
    if (u < 1 || u > 3)
        env->ThrowError("vsTEdgeMask: u must be between 1..3.");
    if (v < 1 || v > 3)
        env->ThrowError("vsTEdgeMask: v must be between 1..3.");

    double th[3] = { threshY, threshU, threshV };
    for (int i = 0; i < 3; ++i)
    {
        th[i] *= static_cast<int64_t>(1) << (bits - 8);
        th[i] *= th[i];

        switch (type)
        {
            case TwoPixel: th[i] *= 4; break;
            case FourPixel: th[i] *= 10000; break;
            case SixPixel: th[i] *= 16; break;
        }

        _threshold[i] = llrint(th[i]);
    }

    const int process_planes[3] = { y, u, v };
    for (int i = 0; i < 3; ++i)
    {
        if (vi.IsRGB())
            process[i] = 3;
        else
        {
            switch (process_planes[i])
            {
                case 3: process[i] = 3; break;
                case 2: process[i] = 2; break;
                default: process[i] = 1; break;
            }
        }
    }

    if (vi.IsY() || _threshold[0] == 0 || _threshold[1] == 0 || _threshold[2] == 0 || process[0] != 3 || process[1] != 3 || process[2] != 3)
        _link = LinkNothing;

    if (type == TwoPixel)
        _scale *= 255.0 / 127.5;
    else if (type == FourPixel)
        _scale *= 255.0 / 158.1;
    else if (type == SixPixel)
        _scale *= 0.25;

    const bool sse2 = (!!(env->GetCPUFlags() & CPUF_SSE2) && opt < 0) || opt == 1;

    for (int plane = 0; plane < vi.NumComponents(); ++plane)
    {
        if (sse2)
        {
            if (bits == 8)
            {
                if (_threshold[plane] == 0)
                {
                    switch (type)
                    {
                        case TwoPixel: detect_edges[plane] = detect_edges_sse2<uint8_t, TwoPixel, false>; break;
                        case FourPixel: detect_edges[plane] = detect_edges_sse2<uint8_t, FourPixel, false>; break;
                        case SixPixel: detect_edges[plane] = detect_edges_sse2<uint8_t, SixPixel, false>; break;
                    }
                }
                else
                {
                    switch (type)
                    {
                        case TwoPixel: detect_edges[plane] = detect_edges_sse2<uint8_t, TwoPixel, true>; break;
                        case FourPixel: detect_edges[plane] = detect_edges_sse2<uint8_t, FourPixel, true>; break;
                        case SixPixel: detect_edges[plane] = detect_edges_sse2<uint8_t, SixPixel, true>; break;
                    }
                }
            }
            else
            {
                if (_threshold[plane] == 0)
                {
                    switch (type)
                    {
                        case TwoPixel: detect_edges[plane] = detect_edges_sse2<uint16_t, TwoPixel, false>; break;
                        case FourPixel: detect_edges[plane] = detect_edges_sse2<uint16_t, FourPixel, false>; break;
                        case SixPixel: detect_edges[plane] = detect_edges_sse2<uint16_t, SixPixel, false>; break;
                    }
                }
                else
                {
                    switch (type)
                    {
                        case TwoPixel: detect_edges[plane] = detect_edges_sse2<uint16_t, TwoPixel, true>; break;
                        case FourPixel: detect_edges[plane] = detect_edges_sse2<uint16_t, FourPixel, true>; break;
                        case SixPixel: detect_edges[plane] = detect_edges_sse2<uint16_t, SixPixel, true>; break;
                    }
                }
            }
        }
        else
        {
            if (bits == 8)
            {
                if (_threshold[plane] == 0)
                {
                    switch (type)
                    {
                        case TwoPixel: detect_edges[plane] = detect_edges_scalar<uint8_t, TwoPixel, false>; break;
                        case FourPixel: detect_edges[plane] = detect_edges_scalar<uint8_t, FourPixel, false>; break;
                        case SixPixel: detect_edges[plane] = detect_edges_scalar<uint8_t, SixPixel, false>; break;
                    }
                }
                else
                {
                    switch (type)
                    {
                        case TwoPixel: detect_edges[plane] = detect_edges_scalar<uint8_t, TwoPixel, true>; break;
                        case FourPixel: detect_edges[plane] = detect_edges_scalar<uint8_t, FourPixel, true>; break;
                        case SixPixel: detect_edges[plane] = detect_edges_scalar<uint8_t, SixPixel, true>; break;
                    }
                }
            }
            else
            {
                if (_threshold[plane] == 0)
                {
                    switch (type)
                    {
                        case TwoPixel: detect_edges[plane] = detect_edges_scalar<uint16_t, TwoPixel, false>; break;
                        case FourPixel: detect_edges[plane] = detect_edges_scalar<uint16_t, FourPixel, false>; break;
                        case SixPixel: detect_edges[plane] = detect_edges_scalar<uint16_t, SixPixel, false>; break;
                    }
                }
                else
                {
                    switch (type)
                    {
                        case TwoPixel: detect_edges[plane] = detect_edges_scalar<uint16_t, TwoPixel, true>; break;
                        case FourPixel: detect_edges[plane] = detect_edges_scalar<uint16_t, FourPixel, true>; break;
                        case SixPixel: detect_edges[plane] = detect_edges_scalar<uint16_t, SixPixel, true>; break;
                    }
                }
            }
        }

        if (subsw == 0)
        {
            if (subsh == 0)
            {
                if (bits == 8)
                {
                    if (_link == LinkChromaToLuma)
                        link_planes = link_planes_444_scalar<uint8_t, LinkChromaToLuma>;
                    else if (_link == LinkEverything)
                        link_planes = link_planes_444_scalar<uint8_t, LinkEverything>;
                }
                else
                {
                    if (_link == LinkChromaToLuma)
                        link_planes = link_planes_444_scalar<uint16_t, LinkChromaToLuma>;
                    else if (_link == LinkEverything)
                        link_planes = link_planes_444_scalar<uint16_t, LinkEverything>;
                }
            }
            else
            {
                if (bits == 8)
                {
                    if (_link == LinkChromaToLuma)
                        link_planes = link_planes_440_scalar<uint8_t, LinkChromaToLuma>;
                    else if (_link == LinkEverything)
                        link_planes = link_planes_440_scalar<uint8_t, LinkEverything>;
                }
                else
                {
                    if (_link == LinkChromaToLuma)
                        link_planes = link_planes_440_scalar<uint16_t, LinkChromaToLuma>;
                    else if (_link == LinkEverything)
                        link_planes = link_planes_440_scalar<uint16_t, LinkEverything>;
                }
            }
        }
        else {
            if (subsh == 0)
            {
                if (bits == 8) {
                    if (_link == LinkChromaToLuma)
                        link_planes = link_planes_422_scalar<uint8_t, LinkChromaToLuma>;
                    else if (_link == LinkEverything)
                        link_planes = link_planes_422_scalar<uint8_t, LinkEverything>;
                }
                else
                {
                    if (_link == LinkChromaToLuma)
                        link_planes = link_planes_422_scalar<uint16_t, LinkChromaToLuma>;
                    else if (_link == LinkEverything)
                        link_planes = link_planes_422_scalar<uint16_t, LinkEverything>;
                }
            }
            else
            {
                if (bits == 8)
                {
                    if (_link == LinkChromaToLuma)
                        link_planes = link_planes_420_scalar<uint8_t, LinkChromaToLuma>;
                    else if (_link == LinkEverything)
                        link_planes = link_planes_420_scalar<uint8_t, LinkEverything>;
                }
                else
                {
                    if (_link == LinkChromaToLuma)
                        link_planes = link_planes_420_scalar<uint16_t, LinkChromaToLuma>;
                    else if (_link == LinkEverything)
                        link_planes = link_planes_420_scalar<uint16_t, LinkEverything>;
                }
            }
        }
    }

    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; };
}

PVideoFrame __stdcall vsTEdgeMask::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = (has_at_least_v8) ? env->NewVideoFrameP(vi, &src) : env->NewVideoFrame(vi);

    const int pixel_max = (1 << vi.BitsPerComponent()) - 1;

    const int planes_y[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    const int planes_r[3] = { PLANAR_R, PLANAR_G, PLANAR_B };
    const int* planes = (vi.IsRGB()) ? planes_r : planes_y;
    const int planecount = std::min(vi.NumComponents(), 3);
    for (int i = 0; i < planecount; ++i)
    {
        if (process[i] != 1)
        {
            const int plane = planes[i];
            int stride = src->GetPitch(plane);
            int dst_stride = dst->GetPitch(plane);
            int width = src->GetRowSize(plane);
            const int height = src->GetHeight(plane);
            const uint8_t* srcp = src->GetReadPtr(plane);
            uint8_t* dstp = dst->GetWritePtr(plane);

            if (process[i] == 3)
                detect_edges[i](srcp, dstp, stride, dst_stride, width, height, _threshold[i], _scale, pixel_max);
            else if (process[i] == 2)
                env->BitBlt(dstp, dst_stride, srcp, stride, width, height);
        }
    }

    if (_link)
    {
        link_planes(
            dst->GetWritePtr(planes[0]),
            dst->GetWritePtr(planes[1]),
            dst->GetWritePtr(planes[2]),
            dst->GetPitch(planes[0]),
            dst->GetPitch(planes[1]),
            dst->GetRowSize(planes[0]),
            dst->GetHeight(planes[0]),
            pixel_max);
    }

    return dst;
}

AVSValue __cdecl Create_vsTEdgeMask(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    PClip clip = args[0].AsClip();
    const VideoInfo& vi = clip->GetVideoInfo();
    int link = (vi.IsRGB()) ? args[5].AsInt(2) : args[5].AsInt(1);

    return new vsTEdgeMask(
        clip,
        args[1].AsFloat(8.0),
        args[2].AsFloat(8.0),
        args[3].AsFloat(8.0),
        args[4].AsInt(FourPixel),
        link,
        args[6].AsFloat(1.0),
        args[7].AsInt(3),
        args[8].AsInt(3),
        args[9].AsInt(3),
        args[10].AsFloat(-1),
        env);
}

const AVS_Linkage* AVS_linkage;

extern "C" __declspec(dllexport)
const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    env->AddFunction("vsTEdgeMask", "c[threshY]f[threshU]f[threshV]f[type]i[link]i[scale]f[y]i[u]i[v]i[opt]i", Create_vsTEdgeMask, 0);

    return "vsTEdgeMask";
}
