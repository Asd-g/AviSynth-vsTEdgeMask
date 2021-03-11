#include "vsTEdgeMask.h"

#include <emmintrin.h>
#include <memory>

#if defined(_WIN32)
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

template<typename PixelType, MaskTypes type, bool binarize>
static FORCE_INLINE void detect_edges_uint8_mmword_sse2(const PixelType* srcp, PixelType* dstp, int x, int stride, const __m128i& threshold, float scale)
{
    __m128i gx, gy;
    __m128 divisor;

    __m128i top = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*) & srcp[x - stride]), _mm_setzero_si128());
    __m128i left = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*) & srcp[x - 1]), _mm_setzero_si128());
    __m128i right = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*) & srcp[x + 1]), _mm_setzero_si128());
    __m128i bottom = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*) & srcp[x + stride]), _mm_setzero_si128());

    if (type == TwoPixel)
    {
        gx = _mm_sub_epi16(right, left);
        gy = _mm_sub_epi16(top, bottom);
        divisor = _mm_set1_ps(0.25f);
    }
    else if (type == FourPixel)
    {
        __m128i top2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*) & srcp[x - 2 * stride]), _mm_setzero_si128());
        __m128i left2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*) & srcp[x - 2]), _mm_setzero_si128());
        __m128i right2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*) & srcp[x + 2]), _mm_setzero_si128());
        __m128i bottom2 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*) & srcp[x + 2 * stride]), _mm_setzero_si128());

        gx = _mm_add_epi16(_mm_mullo_epi16(_mm_set1_epi16(12),
            _mm_sub_epi16(left2, right2)),
            _mm_mullo_epi16(_mm_set1_epi16(74),
                _mm_sub_epi16(right, left)));

        gy = _mm_add_epi16(_mm_mullo_epi16(_mm_set1_epi16(12),
            _mm_sub_epi16(bottom2, top2)),
            _mm_mullo_epi16(_mm_set1_epi16(74),
                _mm_sub_epi16(top, bottom)));

        divisor = _mm_set1_ps(0.0001f);
    }
    else if (type == SixPixel)
    {
        __m128i top_left = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*) & srcp[x - stride - 1]),
            _mm_setzero_si128());
        __m128i top_right = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*) & srcp[x - stride + 1]),
            _mm_setzero_si128());
        __m128i bottom_left = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*) & srcp[x + stride - 1]),
            _mm_setzero_si128());
        __m128i bottom_right = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*) & srcp[x + stride + 1]),
            _mm_setzero_si128());

        __m128i sub_right_left = _mm_sub_epi16(right, left);
        __m128i sub_bottom_top = _mm_sub_epi16(bottom, top);

        gx = _mm_add_epi16(_mm_add_epi16(sub_right_left, sub_right_left),
            _mm_add_epi16(_mm_sub_epi16(top_right, top_left),
                _mm_sub_epi16(bottom_right, bottom_left)));

        gy = _mm_add_epi16(_mm_add_epi16(sub_bottom_top, sub_bottom_top),
            _mm_add_epi16(_mm_sub_epi16(bottom_left, top_left),
                _mm_sub_epi16(bottom_right, top_right)));

        divisor = _mm_set1_ps(1.0f);
    }

    __m128i gx_gy_lo = _mm_unpacklo_epi16(gx, gy);
    __m128i gx_gy_hi = _mm_unpackhi_epi16(gx, gy);
    __m128i sum_squares_lo = _mm_madd_epi16(gx_gy_lo, gx_gy_lo);
    __m128i sum_squares_hi = _mm_madd_epi16(gx_gy_hi, gx_gy_hi);

    __m128i output;

    if (binarize)
    {
        output = _mm_packs_epi16(_mm_packs_epi32(_mm_cmpgt_epi32(sum_squares_lo, threshold),
            _mm_cmpgt_epi32(sum_squares_hi, threshold)),
            _mm_setzero_si128());
    }
    else
    {
        __m128 output_lo = _mm_add_ps(_mm_mul_ps(_mm_sqrt_ps(_mm_mul_ps(_mm_cvtepi32_ps(sum_squares_lo),
            divisor)),
            _mm_set1_ps(scale)),
            _mm_set1_ps(0.5f));

        __m128 output_hi = _mm_add_ps(_mm_mul_ps(_mm_sqrt_ps(_mm_mul_ps(_mm_cvtepi32_ps(sum_squares_hi),
            divisor)),
            _mm_set1_ps(scale)),
            _mm_set1_ps(0.5f));

        output = _mm_packus_epi16(_mm_packs_epi32(_mm_cvttps_epi32(output_lo),
            _mm_cvttps_epi32(output_hi)),
            _mm_setzero_si128());
    }

    _mm_storel_epi64((__m128i*) & dstp[x], output);
}


template<typename PixelType, MaskTypes type, bool binarize>
static FORCE_INLINE void detect_edges_uint16_mmword_sse2(const PixelType* srcp, PixelType* dstp, int x, int stride, const __m128& threshold, float scale, int pixel_max)
{
    __m128 gx, gy, divisor;

    __m128i top = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*) & srcp[x - stride]), _mm_setzero_si128());
    __m128i left = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*) & srcp[x - 1]), _mm_setzero_si128());
    __m128i right = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*) & srcp[x + 1]), _mm_setzero_si128());
    __m128i bottom = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*) & srcp[x + stride]), _mm_setzero_si128());

    if (type == TwoPixel)
    {
        gx = _mm_cvtepi32_ps(_mm_sub_epi32(right, left));
        gy = _mm_cvtepi32_ps(_mm_sub_epi32(top, bottom));
        divisor = _mm_set1_ps(0.25f);
    }
    else if (type == FourPixel)
    {
        __m128i top2 = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*) & srcp[x - 2 * stride]), _mm_setzero_si128());
        __m128i left2 = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*) & srcp[x - 2]), _mm_setzero_si128());
        __m128i right2 = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*) & srcp[x + 2]), _mm_setzero_si128());
        __m128i bottom2 = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*) & srcp[x + 2 * stride]), _mm_setzero_si128());

        __m128i sub_left2_right2 = _mm_sub_epi32(left2, right2);
        __m128i sub_right_left = _mm_sub_epi32(right, left);

        gx = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(sub_left2_right2),
            _mm_set1_ps(12.0f)),
            _mm_mul_ps(_mm_cvtepi32_ps(sub_right_left),
                _mm_set1_ps(74.0f)));

        __m128i sub_bottom2_top2 = _mm_sub_epi32(bottom2, top2);
        __m128i sub_top_bottom = _mm_sub_epi32(top, bottom);

        gy = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(sub_bottom2_top2),
            _mm_set1_ps(12.0f)),
            _mm_mul_ps(_mm_cvtepi32_ps(sub_top_bottom),
                _mm_set1_ps(74.0f)));

        divisor = _mm_set1_ps(0.0001f);
    }
    else if (type == SixPixel)
    {
        __m128i top_left = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*) & srcp[x - stride - 1]), _mm_setzero_si128());
        __m128i top_right = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*) & srcp[x - stride + 1]), _mm_setzero_si128());
        __m128i bottom_left = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*) & srcp[x + stride - 1]), _mm_setzero_si128());
        __m128i bottom_right = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*) & srcp[x + stride + 1]), _mm_setzero_si128());

        __m128i sub_right_left = _mm_sub_epi32(right, left);
        __m128i sub_bottom_top = _mm_sub_epi32(bottom, top);

        gx = _mm_cvtepi32_ps(_mm_add_epi32(_mm_add_epi32(sub_right_left, sub_right_left),
            _mm_add_epi32(_mm_sub_epi32(top_right, top_left),
                _mm_sub_epi32(bottom_right, bottom_left))));

        gy = _mm_cvtepi32_ps(_mm_add_epi32(_mm_add_epi32(sub_bottom_top, sub_bottom_top),
            _mm_add_epi32(_mm_sub_epi32(bottom_left, top_left),
                _mm_sub_epi32(bottom_right, top_right))));

        divisor = _mm_set1_ps(1.0f);
    }

    __m128 sum_squares = _mm_add_ps(_mm_mul_ps(gx, gx), _mm_mul_ps(gy, gy));

    __m128i output;

    if (binarize)
    {
        output = _mm_packs_epi32(_mm_castps_si128(_mm_cmpnle_ps(sum_squares, threshold)),
            _mm_setzero_si128());
        output = _mm_and_si128(output,
            _mm_set1_epi16(pixel_max));
    }
    else
    {
        output = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(_mm_sqrt_ps(_mm_mul_ps(sum_squares, divisor)),
            _mm_set1_ps(scale)),
            _mm_set1_ps(-32767.5f))); // 0.5 for rounding and -32768 for packing

        output = _mm_add_epi16(_mm_min_epi16(_mm_packs_epi32(output, output),
            _mm_set1_epi16(pixel_max - 32768)),
            _mm_set1_epi16(32768));
    }

    _mm_storel_epi64((__m128i*) & dstp[x], output);
}


template<typename PixelType, MaskTypes type, bool binarize>
void detect_edges_sse2(const uint8_t* srcp8, uint8_t* dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max)
{
    const PixelType* srcp = (const PixelType*)srcp8;
    PixelType* dstp = (PixelType*)dstp8;
    stride /= sizeof(PixelType);
    dst_stride /= sizeof(PixelType);
    width /= sizeof(PixelType);

    __m128i threshold_epi32 = _mm_set1_epi32((int)threshold64);
    __m128 threshold_ps = _mm_set1_ps((float)threshold64);

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
            if (sizeof(PixelType) == 1)
                detect_edges_uint8_mmword_sse2<PixelType, type, binarize>(srcp, dstp, x, stride, threshold_epi32, scale);
            else
                detect_edges_uint16_mmword_sse2<PixelType, type, binarize>(srcp, dstp, x, stride, threshold_ps, scale, pixel_max);
        }

        if (width > width_simd)
        {
            if (sizeof(PixelType) == 1)
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

template void detect_edges_sse2<uint8_t, TwoPixel, false>(const uint8_t* srcp8, uint8_t* dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max);
template void detect_edges_sse2<uint8_t, FourPixel, false>(const uint8_t* srcp8, uint8_t* dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max);
template void detect_edges_sse2<uint8_t, SixPixel, false>(const uint8_t* srcp8, uint8_t* dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max);

template void detect_edges_sse2<uint8_t, TwoPixel, true>(const uint8_t* srcp8, uint8_t* dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max);
template void detect_edges_sse2<uint8_t, FourPixel, true>(const uint8_t* srcp8, uint8_t* dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max);
template void detect_edges_sse2<uint8_t, SixPixel, true>(const uint8_t* srcp8, uint8_t* dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max);

template void detect_edges_sse2<uint16_t, TwoPixel, false>(const uint8_t* srcp8, uint8_t* dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max);
template void detect_edges_sse2<uint16_t, FourPixel, false>(const uint8_t* srcp8, uint8_t* dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max);
template void detect_edges_sse2<uint16_t, SixPixel, false>(const uint8_t* srcp8, uint8_t* dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max);

template void detect_edges_sse2<uint16_t, TwoPixel, true>(const uint8_t* srcp8, uint8_t* dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max);
template void detect_edges_sse2<uint16_t, FourPixel, true>(const uint8_t* srcp8, uint8_t* dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max);
template void detect_edges_sse2<uint16_t, SixPixel, true>(const uint8_t* srcp8, uint8_t* dstp8, int stride, int dst_stride, int width, int height, int64_t threshold64, float scale, int pixel_max);
