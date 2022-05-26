#include <memory>

#include "VCL2/vectorclass.h"
#include "vsTEdgeMask.h"


template<typename PixelType, MaskTypes type, bool binarize>
static AVS_FORCEINLINE void detect_edges_uint8_mmword_sse2(const PixelType* srcp, PixelType* __restrict dstp, int x, int stride, const Vec4i& threshold, float scale) noexcept
{
    Vec8s gx, gy;
    Vec4f divisor;

    const auto top = Vec8s().load_8uc(srcp + (x - stride));
    const auto left = Vec8s().load_8uc(srcp + (x - 1));
    const auto right = Vec8s().load_8uc(srcp + (x + 1));
    const auto bottom = Vec8s().load_8uc(srcp + (x + stride));

    if constexpr (type == TwoPixel)
    {
        gx = right - left;
        gy = top - bottom;
        divisor = 0.25f;
    }
    else if constexpr (type == FourPixel)
    {
        const auto top2 = Vec8s().load_8uc(srcp + (x - 2 * stride));
        const auto left2 = Vec8s().load_8uc(srcp + (x - 2));
        const auto right2 = Vec8s().load_8uc(srcp + (x + 2));
        const auto bottom2 = Vec8s().load_8uc(srcp + (x + 2 * stride));

        gx = 12 * (left2 - right2) + 74 * (right - left);

        gy = 12 * (bottom2 - top2) + 74 * (top - bottom);

        divisor = 0.0001f;
    }
    else if constexpr (type == SixPixel)
    {
        const auto top_left = Vec8s().load_8uc(srcp + (x - stride - 1));
        const auto top_right = Vec8s().load_8uc(srcp + (x - stride + 1));
        const auto bottom_left = Vec8s().load_8uc(srcp + (x + stride - 1));
        const auto bottom_right = Vec8s().load_8uc(srcp + (x + stride + 1));

        const auto sub_right_left = right - left;
        const auto sub_bottom_top = bottom - top;

        gx = sub_right_left + sub_right_left + (top_right - top_left) + (bottom_right - bottom_left);

        gy = sub_bottom_top + sub_bottom_top + (bottom_left - top_left) + (bottom_right - top_right);

        divisor = 1.0f;
    }

    const auto gx_gy_lo = blend8<0, 8, 1, 9, 2, 10, 3, 11>(gx, gy);
    const auto gx_gy_hi = blend8<4, 12, 5, 13, 6, 14, 7, 15>(gx, gy);

    if constexpr (binarize)
        compress_saturated(compress_saturated(madd(gx_gy_lo, gx_gy_lo) > threshold, madd(gx_gy_hi, gx_gy_hi) > threshold), zero_si128()).storel(dstp + x);
    else
        compress_saturated(compress_saturated_s2u(truncatei(sqrt(to_float(madd(gx_gy_lo, gx_gy_lo)) * divisor) * scale + 0.5f),
            truncatei(sqrt(to_float(madd(gx_gy_hi, gx_gy_hi)) * divisor) * scale + 0.5f)), zero_si128()).storel(dstp + x);
}


template<typename PixelType, MaskTypes type, bool binarize>
static AVS_FORCEINLINE void detect_edges_uint16_mmword_sse2(const PixelType* srcp, PixelType* __restrict dstp, int x, int stride, const Vec4f& threshold, float scale, int pixel_max) noexcept
{
    Vec4f gx, gy, divisor;

    const auto top = Vec4i().load_4us(srcp + (x - stride));
    const auto left = Vec4i().load_4us(srcp + (x - 1));
    const auto right = Vec4i().load_4us(srcp + (x + 1));
    const auto bottom = Vec4i().load_4us(srcp + (x + stride));

    if constexpr (type == TwoPixel)
    {
        gx = to_float(right - left);
        gy = to_float(top - bottom);
        divisor = 0.25f;
    }
    else if constexpr (type == FourPixel)
    {
        const auto top2 = Vec4i().load_4us(srcp + (x - 2 * stride));
        const auto left2 = Vec4i().load_4us(srcp + (x - 2));
        const auto right2 = Vec4i().load_4us(srcp + (x + 2));
        const auto bottom2 = Vec4i().load_4us(srcp + (x + 2 * stride));

        const auto sub_left2_right2 = left2 - right2;
        const auto sub_right_left = right - left;

        gx = to_float(sub_left2_right2) * 12.0f + to_float(sub_right_left) * 74.0f;

        const auto sub_bottom2_top2 = bottom2 - top2;
        const auto sub_top_bottom = top - bottom;

        gy = to_float(sub_bottom2_top2) * 12.0f + to_float(sub_top_bottom) * 74.0f;

        divisor = 0.0001f;
    }
    else if constexpr (type == SixPixel)
    {
        const auto top_left = Vec4i().load_4us(srcp + (x - stride - 1));
        const auto top_right = Vec4i().load_4us(srcp + (x - stride + 1));
        const auto bottom_left = Vec4i().load_4us(srcp + (x + stride - 1));
        const auto bottom_right = Vec4i().load_4us(srcp + (x + stride + 1));

        const auto sub_right_left = right - left;
        const auto sub_bottom_top = bottom - top;

        gx = to_float((sub_right_left + sub_right_left) + (top_right - top_left) + (bottom_right - bottom_left));

        gy = to_float(sub_bottom_top + sub_bottom_top + (bottom_left - top_left) + (bottom_right - top_right));

        divisor = 1.0f;
    }

    if constexpr (binarize)
        (compress_saturated(Vec4i((gx * gx + gy * gy) > threshold), zero_si128()) & pixel_max).storel(dstp + x);
    else
    {
        auto output = truncatei(sqrt((gx * gx + gy * gy) * divisor) * scale + -32767.5f); // 0.5 for rounding and -32768 for packing

        (min(compress_saturated(output, output), pixel_max - 32768) + 32768).storel(dstp + x);
    }
}


template<typename PixelType, MaskTypes type, bool binarize>
void detect_edges_sse2(const uint8_t* srcp8, uint8_t* __restrict dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max) noexcept
{
    const PixelType* srcp = reinterpret_cast<const PixelType*>(srcp8);
    PixelType* __restrict dstp = reinterpret_cast<PixelType*>(dstp8);
    stride /= sizeof(PixelType);
    dst_stride /= sizeof(PixelType);
    width /= sizeof(PixelType);

    const Vec4i threshold_epi32 = static_cast<int>(threshold64);
    const Vec4f threshold_ps = static_cast<float>(threshold64);

    // Number of pixels to skip at the edges of the image.
    const int skip = type == FourPixel ? 2 : 1;

    const int pixels_per_iteration = 8 / sizeof(PixelType);

    const int width_simd = (width - 2 * skip) / pixels_per_iteration * pixels_per_iteration + 2 * skip;

    for (int y = 0; y < skip; ++y)
    {
        memset(dstp, 0, width * sizeof(PixelType));

        srcp += stride;
        dstp += dst_stride;
    }

    for (int y = skip; y < height - skip; ++y)
    {
        memset(dstp, 0, skip * sizeof(PixelType));

        for (int x = skip; x < width_simd - skip; x += pixels_per_iteration)
        {
            if constexpr (std::is_same_v<PixelType, uint8_t>)
                detect_edges_uint8_mmword_sse2<PixelType, type, binarize>(srcp, dstp, x, stride, threshold_epi32, scale);
            else
                detect_edges_uint16_mmword_sse2<PixelType, type, binarize>(srcp, dstp, x, stride, threshold_ps, scale, pixel_max);
        }

        if (width > width_simd)
        {
            if constexpr (std::is_same_v<PixelType, uint8_t>)
                detect_edges_uint8_mmword_sse2<PixelType, type, binarize>(srcp, dstp, width - skip - pixels_per_iteration, stride, threshold_epi32, scale);
            else
                detect_edges_uint16_mmword_sse2<PixelType, type, binarize>(srcp, dstp, width - skip - pixels_per_iteration, stride, threshold_ps, scale, pixel_max);
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

template void detect_edges_sse2<uint8_t, TwoPixel, false>(const uint8_t* srcp8, uint8_t* __restrict dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max) noexcept;
template void detect_edges_sse2<uint8_t, FourPixel, false>(const uint8_t* srcp8, uint8_t* __restrict dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max) noexcept;
template void detect_edges_sse2<uint8_t, SixPixel, false>(const uint8_t* srcp8, uint8_t* __restrict dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max) noexcept;

template void detect_edges_sse2<uint8_t, TwoPixel, true>(const uint8_t* srcp8, uint8_t* __restrict dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max) noexcept;
template void detect_edges_sse2<uint8_t, FourPixel, true>(const uint8_t* srcp8, uint8_t* __restrict dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max) noexcept;
template void detect_edges_sse2<uint8_t, SixPixel, true>(const uint8_t* srcp8, uint8_t* __restrict dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max) noexcept;

template void detect_edges_sse2<uint16_t, TwoPixel, false>(const uint8_t* srcp8, uint8_t* __restrict dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max) noexcept;
template void detect_edges_sse2<uint16_t, FourPixel, false>(const uint8_t* srcp8, uint8_t* __restrict dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max) noexcept;
template void detect_edges_sse2<uint16_t, SixPixel, false>(const uint8_t* srcp8, uint8_t* __restrict dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max) noexcept;

template void detect_edges_sse2<uint16_t, TwoPixel, true>(const uint8_t* srcp8, uint8_t* __restrict dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max) noexcept;
template void detect_edges_sse2<uint16_t, FourPixel, true>(const uint8_t* srcp8, uint8_t* __restrict dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max) noexcept;
template void detect_edges_sse2<uint16_t, SixPixel, true>(const uint8_t* srcp8, uint8_t* __restrict dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max) noexcept;
