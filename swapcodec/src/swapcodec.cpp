// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "swapcodec.h"

#include "apex_memmove/apex_memmove.h"
#include "apex_memmove/apex_memmove.c"

#include <intrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#ifdef SSSE3
#include <tmmintrin.h>
#endif

#pragma warning(push, 0)
#include "mango/core/thread.hpp"
#pragma warning(pop)

using namespace swapcodec;

//////////////////////////////////////////////////////////////////////////

#define DCT_PER_BLOCK_SIZE 128

//////////////////////////////////////////////////////////////////////////

void swapcodec::swapMemcpy(OUT void * pDestination, IN const void * pSource, const size_t size)
{
  apex_memcpy(pDestination, pSource, size);
}

void swapcodec::swapMemmove(OUT void * pDestination, IN_OUT void * pSource, const size_t size)
{
  apex_memmove(pDestination, pSource, size);
}

//////////////////////////////////////////////////////////////////////////

swapResult swapEncodeFrameYUV420(IN uint8_t *pImage, OUT uint8_t *pUncompressedData, const size_t resX, const size_t resY, mango::ConcurrentQueue *pQueue);
swapResult swapDecodeFrameYUV420(IN uint8_t *pUncompressedData, OUT uint8_t *pImage, const size_t resX, const size_t resY, mango::ConcurrentQueue *pQueue);
swapResult swapCompressData(IN uint8_t *pData, OUT uint8_t *pCompressedData, IN_OUT size_t *pCompressedDataCapacity, OUT size_t *pCompressedDataLength);
swapResult swapDecompressData(IN uint8_t *pCompressedData, const size_t compressedDataLength, OUT uint8_t *pUncompressedData, IN_OUT size_t *pUncompressedDataCompressedLength, OUT size_t *pUncompressedDataLength);

//////////////////////////////////////////////////////////////////////////

swapEncoder * swapcodec::swapEncoder::Create(const std::string &filename, const size_t resX, const size_t resY)
{
  swapEncoder *pEncoder = nullptr;

  if ((resX & 63) != 0 || (resY & 63) != 0)
    goto epilogue;

  pEncoder = new swapEncoder();
  
  if (pEncoder == nullptr)
    goto epilogue;

  pEncoder->filename = filename;

  pEncoder->resX = resX;
  pEncoder->resY = resY;
  pEncoder->lowResX = resX << 3;
  pEncoder->lowResY = resY << 4;

  pEncoder->pCompressibleData = (uint8_t *)malloc(sizeof(uint8_t) * ((resX >> 3) * ((resY * 3 / 2) >> 3) * DCT_PER_BLOCK_SIZE));

  if (pEncoder->pCompressibleData == nullptr)
    goto epilogue;

  pEncoder->pThreadPool = new mango::ConcurrentQueue();

  if (pEncoder->pThreadPool == nullptr)
    goto epilogue;

  return pEncoder;

epilogue:
  if (pEncoder)
    delete pEncoder;

  return nullptr;
}

swapcodec::swapEncoder::~swapEncoder()
{
  if (pCompressibleData)
    free(pCompressibleData);

  if (pThreadPool)
    delete (mango::ConcurrentQueue *)pThreadPool;
}

swapResult swapcodec::swapEncoder::AddFrameYUV420(IN_OUT uint8_t *pFrameData)
{
  swapResult result = sR_Success;

  if (sR_Success != (result = swapEncodeFrameYUV420(pFrameData, pCompressibleData, resX, resY, (mango::ConcurrentQueue *)pThreadPool)))
    goto epilogue;

  //swapMemcpy(pFrameData, pCompressibleData, resX * resY * 3 / 2);

  if (sR_Success != (result = swapDecodeFrameYUV420(pCompressibleData, pFrameData, resX, resY, (mango::ConcurrentQueue *)pThreadPool)))
    goto epilogue;

epilogue:
  return result;
}

//////////////////////////////////////////////////////////////////////////

const uint8_t zigzag_table[] =
{
  0,  1,   5,  6, 14, 15, 27, 28,
  2,  4,   7, 13, 16, 26, 29, 42,
  3,  8,  12, 17, 25, 30, 41, 43,
  9,  11, 18, 24, 31, 40, 44, 53,
  10, 19, 23, 32, 39, 45, 52, 54,
  20, 22, 33, 38, 46, 51, 55, 60,
  21, 34, 37, 47, 50, 56, 59, 61,
  35, 36, 48, 49, 57, 58, 62, 63
};

const uint8_t luminance_quant_table[] =
{
  16, 11, 10, 16,  24,  40,  51,  61,
  12, 12, 14, 19,  26,  58,  60,  55,
  14, 13, 16, 24,  40,  57,  69,  56,
  14, 17, 22, 29,  51,  87,  80,  62,
  18, 22, 37, 56,  68, 109, 103,  77,
  24, 35, 55, 64,  81, 104, 113,  92,
  49, 64, 78, 87, 103, 121, 120, 101,
  72, 92, 95, 98, 112, 100, 103,  99
};

const uint8_t chrominance_quant_table[] =
{
  17, 18, 24, 47, 99, 99, 99, 99,
  18, 21, 26, 66, 99, 99, 99, 99,
  24, 26, 56, 99, 99, 99, 99, 99,
  47, 66, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99
};

void swapInitDctQuantizationTables(uint32_t quality, uint8_t *pLqt, uint8_t *pCqt, uint16_t *pILqt, uint16_t *pICqt)
{
  quality = std::min(quality, 1024u) * 16;

  for (size_t i = 0; i < 64; ++i)
  {
    uint16_t index = zigzag_table[i];
    uint32_t value;

    // luminance quantization table * quality factor
    value = luminance_quant_table[i] * quality;
    value = (value + 0x200) >> 10;

    if (value < 2)
      value = 2;
    else if (value > 255)
      value = 255;

    pLqt[index] = (uint8_t)value;
    pILqt[i] = static_cast<uint16_t>(0x8000 / value);

    // chrominance quantization table * quality factor
    value = chrominance_quant_table[i] * quality;
    value = (value + 0x200) >> 10;

    if (value < 2)
      value = 2;
    else if (value > 255)
      value = 255;

    pCqt[index] = (uint8_t)value;
    pICqt[i] = static_cast<uint16_t>(0x8000 / value);
  }
}

void slapDCT(int16_t * pDestination, int16_t * pData, const uint16_t * pQuantizationTable)
{
  const uint16_t c1 = 1420;  // cos  PI/16 * root(2)
  const uint16_t c2 = 1338;  // cos  PI/8  * root(2)
  const uint16_t c3 = 1204;  // cos 3PI/16 * root(2)
  const uint16_t c5 = 805;   // cos 5PI/16 * root(2)
  const uint16_t c6 = 554;   // cos 3PI/8  * root(2)
  const uint16_t c7 = 283;   // cos 7PI/16 * root(2)

  for (int32_t i = 0; i < 8; ++i)
  {
    int32_t x8 = pData[0] + pData[7];
    int32_t x0 = pData[0] - pData[7];
    int32_t x7 = pData[1] + pData[6];
    int32_t x1 = pData[1] - pData[6];
    int32_t x6 = pData[2] + pData[5];
    int32_t x2 = pData[2] - pData[5];
    int32_t x5 = pData[3] + pData[4];
    int32_t x3 = pData[3] - pData[4];
    int32_t x4 = x8 + x5;

    x8 = x8 - x5;
    x5 = x7 + x6;
    x7 = x7 - x6;

    pData[0] = int16_t(x4 + x5);
    pData[4] = int16_t(x4 - x5);
    pData[2] = int16_t((x8 * c2 + x7 * c6) >> 10);
    pData[6] = int16_t((x8 * c6 - x7 * c2) >> 10);
    pData[7] = int16_t((x0 * c7 - x1 * c5 + x2 * c3 - x3 * c1) >> 10);
    pData[5] = int16_t((x0 * c5 - x1 * c1 + x2 * c7 + x3 * c3) >> 10);
    pData[3] = int16_t((x0 * c3 - x1 * c7 - x2 * c1 - x3 * c5) >> 10);
    pData[1] = int16_t((x0 * c1 + x1 * c3 + x2 * c5 + x3 * c7) >> 10);
    pData += 8;
  }

  pData -= 64;

  for (int32_t i = 0; i < 8; ++i)
  {
    int32_t x8 = pData[i + 0] + pData[i + 56];
    int32_t x0 = pData[i + 0] - pData[i + 56];
    int32_t x7 = pData[i + 8] + pData[i + 48];
    int32_t x1 = pData[i + 8] - pData[i + 48];
    int32_t x6 = pData[i + 16] + pData[i + 40];
    int32_t x2 = pData[i + 16] - pData[i + 40];
    int32_t x5 = pData[i + 24] + pData[i + 32];
    int32_t x3 = pData[i + 24] - pData[i + 32];
    int32_t x4 = x8 + x5;

    x8 = x8 - x5;
    x5 = x7 + x6;
    x7 = x7 - x6;

    int16_t v0 = int16_t((x4 + x5) >> 3);
    int16_t v4 = int16_t((x4 - x5) >> 3);
    int16_t v2 = int16_t((x8 * c2 + x7 * c6) >> 13);
    int16_t v6 = int16_t((x8 * c6 - x7 * c2) >> 13);
    int16_t v7 = int16_t((x0 * c7 - x1 * c5 + x2 * c3 - x3 * c1) >> 13);
    int16_t v5 = int16_t((x0 * c5 - x1 * c1 + x2 * c7 + x3 * c3) >> 13);
    int16_t v3 = int16_t((x0 * c3 - x1 * c7 - x2 * c1 - x3 * c5) >> 13);
    int16_t v1 = int16_t((x0 * c1 + x1 * c3 + x2 * c5 + x3 * c7) >> 13);

    pDestination[/*zigzag_table[*/i + 0 * 8/*]*/] = int16_t((v0 * pQuantizationTable[i + 0 * 8] + 0x4000) >> 15);
    pDestination[/*zigzag_table[*/i + 1 * 8/*]*/] = int16_t((v1 * pQuantizationTable[i + 1 * 8] + 0x4000) >> 15);
    pDestination[/*zigzag_table[*/i + 2 * 8/*]*/] = int16_t((v2 * pQuantizationTable[i + 2 * 8] + 0x4000) >> 15);
    pDestination[/*zigzag_table[*/i + 3 * 8/*]*/] = int16_t((v3 * pQuantizationTable[i + 3 * 8] + 0x4000) >> 15);
    pDestination[/*zigzag_table[*/i + 4 * 8/*]*/] = int16_t((v4 * pQuantizationTable[i + 4 * 8] + 0x4000) >> 15);
    pDestination[/*zigzag_table[*/i + 5 * 8/*]*/] = int16_t((v5 * pQuantizationTable[i + 5 * 8] + 0x4000) >> 15);
    pDestination[/*zigzag_table[*/i + 6 * 8/*]*/] = int16_t((v6 * pQuantizationTable[i + 6 * 8] + 0x4000) >> 15);
    pDestination[/*zigzag_table[*/i + 7 * 8/*]*/] = int16_t((v7 * pQuantizationTable[i + 7 * 8] + 0x4000) >> 15);
  }
}

void swapFormatMCUBlock(int16_t * pBlock, uint8_t * pInput, int rows, int cols, int incr)
{
  for (int i = 0; i < rows; ++i)
  {
    for (int j = cols; j > 0; --j)
    {
      *pBlock++ = (*pInput++) - 128;
    }

    // replicate last column
    for (int j = 8 - cols; j > 0; --j)
    {
      *pBlock = *(pBlock - 1);
      ++pBlock;
    }

    pInput += incr;
  }

  // replicate last row
  for (int i = 8 - rows; i > 0; --i)
  {
    for (int j = 8; j > 0; --j)
    {
      *pBlock = *(pBlock - 8);
      ++pBlock;
    }
  }
}

//////////////////////////////////////////////////////////////////////////

static const int _izigzag_table_standard[] =
{
  0 , 1 , 8 , 16, 9 , 2 , 3 , 10, 17, 24, 32, 25, 18, 11, 4 , 5 ,
  12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6 , 7 , 14, 21, 28,
  35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
  58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,

  63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
};

static const int _izigzag_table_variant[] =
{
  0,  8,  1,  2,  9, 16, 24, 17, 10,  3,  4, 11, 18, 25, 32, 40,
  33, 26, 19, 12,  5,  6, 13, 20, 27, 34, 41, 48, 56, 49, 42, 35,
  28, 21, 14,  7, 15, 22, 29, 36, 43, 50, 57, 58, 51, 44, 37, 30,
  23, 31, 38, 45, 52, 59, 60, 53, 46, 39, 47, 54, 61, 62, 55, 63,

  63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
};

constexpr int _IDCT_PREC = 12;
constexpr int _IDCT_HALF(int precision) { return (1 << ((precision)-1)); }
constexpr int _IDCT_FIXED(double x) { return int((x * double(1 << _IDCT_PREC) + 0.5)); }

constexpr int _IDCT_M_2_562915447 = _IDCT_FIXED(-2.562915447);
constexpr int _IDCT_M_1_961570560 = _IDCT_FIXED(-1.961570560);
constexpr int _IDCT_M_1_847759065 = _IDCT_FIXED(-1.847759065);
constexpr int _IDCT_M_0_899976223 = _IDCT_FIXED(-0.899976223);
constexpr int _IDCT_M_0_390180644 = _IDCT_FIXED(-0.390180644);
constexpr int _IDCT_P_0_298631336 = _IDCT_FIXED(0.298631336);
constexpr int _IDCT_P_0_541196100 = _IDCT_FIXED(0.541196100);
constexpr int _IDCT_P_0_765366865 = _IDCT_FIXED(0.765366865);
constexpr int _IDCT_P_1_175875602 = _IDCT_FIXED(1.175875602);
constexpr int _IDCT_P_1_501321110 = _IDCT_FIXED(1.501321110);
constexpr int _IDCT_P_2_053119869 = _IDCT_FIXED(2.053119869);
constexpr int _IDCT_P_3_072711026 = _IDCT_FIXED(3.072711026);

// Keep 2 bits of extra precision for the intermediate results
constexpr int _IDCT_COL_NORM = (_IDCT_PREC - 2);
constexpr int _IDCT_COL_BIAS = _IDCT_HALF(_IDCT_COL_NORM);

// Consume 2 bits of an intermediate results precision and 3 bits that were
// produced by `2 * sqrt(8)`. Also normalize to from `-128..127` to `0..255`
constexpr int _IDCT_ROW_NORM = (_IDCT_PREC + 2 + 3);
constexpr int _IDCT_ROW_BIAS = (_IDCT_HALF(_IDCT_ROW_NORM) + (128 << _IDCT_ROW_NORM));

#define _CONST16_SSE2(x, y)  _mm_setr_epi16(x, y, x, y, x, y, x, y)
#define _CONST32_SSE2(x)     _mm_setr_epi32(x, x, x, x)

static const __m128i rot0_0 = _CONST16_SSE2(_IDCT_P_0_541196100, _IDCT_P_0_541196100 + _IDCT_M_1_847759065);
static const __m128i rot0_1 = _CONST16_SSE2(_IDCT_P_0_541196100 + _IDCT_P_0_765366865, _IDCT_P_0_541196100);
static const __m128i rot1_0 = _CONST16_SSE2(_IDCT_P_1_175875602 + _IDCT_M_0_899976223, _IDCT_P_1_175875602);
static const __m128i rot1_1 = _CONST16_SSE2(_IDCT_P_1_175875602, _IDCT_P_1_175875602 + _IDCT_M_2_562915447);
static const __m128i rot2_0 = _CONST16_SSE2(_IDCT_M_1_961570560 + _IDCT_P_0_298631336, _IDCT_M_1_961570560);
static const __m128i rot2_1 = _CONST16_SSE2(_IDCT_M_1_961570560, _IDCT_M_1_961570560 + _IDCT_P_3_072711026);
static const __m128i rot3_0 = _CONST16_SSE2(_IDCT_M_0_390180644 + _IDCT_P_2_053119869, _IDCT_M_0_390180644);
static const __m128i rot3_1 = _CONST16_SSE2(_IDCT_M_0_390180644, _IDCT_M_0_390180644 + _IDCT_P_1_501321110);
static const __m128i colBias = _CONST32_SSE2(_IDCT_COL_BIAS);
static const __m128i rowBias = _CONST32_SSE2(_IDCT_ROW_BIAS);

#define _IDCT_ROTATE_XMM(dst0, dst1, x, y, c0, c1) \
    __m128i c0##_l = _mm_unpacklo_epi16(x, y); \
    __m128i c0##_h = _mm_unpackhi_epi16(x, y); \
    __m128i dst0##_l = _mm_madd_epi16(c0##_l, c0); \
    __m128i dst0##_h = _mm_madd_epi16(c0##_h, c0); \
    __m128i dst1##_l = _mm_madd_epi16(c0##_l, c1); \
    __m128i dst1##_h = _mm_madd_epi16(c0##_h, c1);

// out = in << 12  (in 16-bit, out 32-bit)
#define _IDCT_WIDEN_XMM(dst, in) \
    __m128i dst##_l = _mm_srai_epi32(_mm_unpacklo_epi16(_mm_setzero_si128(), (in)), 4); \
    __m128i dst##_h = _mm_srai_epi32(_mm_unpackhi_epi16(_mm_setzero_si128(), (in)), 4);

// wide add
#define _IDCT_WADD_XMM(dst, a, b) \
    __m128i dst##_l = _mm_add_epi32(a##_l, b##_l); \
    __m128i dst##_h = _mm_add_epi32(a##_h, b##_h);

// wide sub
#define _IDCT_WSUB_XMM(dst, a, b) \
    __m128i dst##_l = _mm_sub_epi32(a##_l, b##_l); \
    __m128i dst##_h = _mm_sub_epi32(a##_h, b##_h);

// butterfly a/b, add bias, then shift by `norm` and pack to 16-bit
#define _IDCT_BFLY_XMM(dst0, dst1, a, b, bias, norm) { \
    __m128i abiased_l = _mm_add_epi32(a##_l, bias); \
    __m128i abiased_h = _mm_add_epi32(a##_h, bias); \
    _IDCT_WADD_XMM(sum, abiased, b) \
    _IDCT_WSUB_XMM(diff, abiased, b) \
    dst0 = _mm_packs_epi32(_mm_srai_epi32(sum_l, norm), _mm_srai_epi32(sum_h, norm)); \
    dst1 = _mm_packs_epi32(_mm_srai_epi32(diff_l, norm), _mm_srai_epi32(diff_h, norm)); \
    }

#define _IDCT_IDCT_PASS_XMM(bias, norm) { \
    _IDCT_ROTATE_XMM(t2e, t3e, v2, v6, rot0_0, rot0_1) \
    __m128i sum04 = _mm_add_epi16(v0, v4); \
    __m128i dif04 = _mm_sub_epi16(v0, v4); \
    _IDCT_WIDEN_XMM(t0e, sum04) \
    _IDCT_WIDEN_XMM(t1e, dif04) \
    _IDCT_WADD_XMM(x0, t0e, t3e) \
    _IDCT_WSUB_XMM(x3, t0e, t3e) \
    _IDCT_WADD_XMM(x1, t1e, t2e) \
    _IDCT_WSUB_XMM(x2, t1e, t2e) \
    _IDCT_ROTATE_XMM(y0o, y2o, v7, v3, rot2_0, rot2_1) \
    _IDCT_ROTATE_XMM(y1o, y3o, v5, v1, rot3_0, rot3_1) \
    __m128i sum17 = _mm_add_epi16(v1, v7); \
    __m128i sum35 = _mm_add_epi16(v3, v5); \
    _IDCT_ROTATE_XMM(y4o,y5o, sum17, sum35, rot1_0, rot1_1) \
    _IDCT_WADD_XMM(x4, y0o, y4o) \
    _IDCT_WADD_XMM(x5, y1o, y5o) \
    _IDCT_WADD_XMM(x6, y2o, y5o) \
    _IDCT_WADD_XMM(x7, y3o, y4o) \
    _IDCT_BFLY_XMM(v0, v7, x0, x7, bias, norm) \
    _IDCT_BFLY_XMM(v1, v6, x1, x6, bias, norm) \
    _IDCT_BFLY_XMM(v2, v5, x2, x5, bias, norm) \
    _IDCT_BFLY_XMM(v3, v4, x3, x4, bias, norm) \
    }

static inline void interleave8(__m128i &a, __m128i &b)
{
  __m128i c = a;
  a = _mm_unpacklo_epi8(a, b);
  b = _mm_unpackhi_epi8(c, b);
}

static inline void interleave16(__m128i &a, __m128i &b)
{
  __m128i c = a;
  a = _mm_unpacklo_epi16(a, b);
  b = _mm_unpackhi_epi16(c, b);
}

void idct_sse2(uint8_t* dest, int stride, const int16_t* src, const uint16_t* qt)
{
  (void)(stride);

  const __m128i* data = reinterpret_cast<const __m128i *>(src);
  const __m128i* qtable = reinterpret_cast<const __m128i *>(qt);

  // Load and dequantize
  __m128i v0 = _mm_mullo_epi16(data[0], qtable[0]);
  __m128i v1 = _mm_mullo_epi16(data[1], qtable[1]);
  __m128i v2 = _mm_mullo_epi16(data[2], qtable[2]);
  __m128i v3 = _mm_mullo_epi16(data[3], qtable[3]);
  __m128i v4 = _mm_mullo_epi16(data[4], qtable[4]);
  __m128i v5 = _mm_mullo_epi16(data[5], qtable[5]);
  __m128i v6 = _mm_mullo_epi16(data[6], qtable[6]);
  __m128i v7 = _mm_mullo_epi16(data[7], qtable[7]);

  // IDCT columns
  _IDCT_IDCT_PASS_XMM(colBias, 10);

  // Transpose
  interleave16(v0, v4);
  interleave16(v2, v6);
  interleave16(v1, v5);
  interleave16(v3, v7);

  interleave16(v0, v2);
  interleave16(v1, v3);
  interleave16(v4, v6);
  interleave16(v5, v7);

  interleave16(v0, v1);
  interleave16(v2, v3);
  interleave16(v4, v5);
  interleave16(v6, v7);

  // IDCT rows
  _IDCT_IDCT_PASS_XMM(rowBias, 17);

  // Pack to 8-bit integers, also saturates the result to 0..255
  __m128i s0 = _mm_packus_epi16(v0, v1);
  __m128i s1 = _mm_packus_epi16(v2, v3);
  __m128i s2 = _mm_packus_epi16(v4, v5);
  __m128i s3 = _mm_packus_epi16(v6, v7);

  // Transpose
  interleave8(s0, s2);
  interleave8(s1, s3);
  interleave8(s0, s1);
  interleave8(s2, s3);
  interleave8(s0, s2);
  interleave8(s1, s3);

  //__m128i x = { 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1 };

  // Store
  __m128i* d = reinterpret_cast<__m128i *>(dest);
  _mm_storeu_si128(d + 0, s0);
  _mm_storeu_si128(d + 1, s2);
  _mm_storeu_si128(d + 2, s1);
  _mm_storeu_si128(d + 3, s3);
}

//////////////////////////////////////////////////////////////////////////

swapResult swapEncodeFrameYUV420(IN uint8_t * pImage, OUT uint8_t * pUncompressedData, const size_t resX, const size_t resY, mango::ConcurrentQueue *pQueue)
{
  swapResult result = sR_Success;

  const size_t blockX = resX >> 3;
  const size_t blockY = resY >> 3;

  uint8_t Lqt[64];
  uint8_t Cqt[64];
  uint16_t ILqt[64];
  uint16_t ICqt[64];
  uint32_t quality = 75;

  swapInitDctQuantizationTables(quality, Lqt, Cqt, ILqt, ICqt);

  for (size_t y = 0; y < blockY; y++)
  {
    pQueue->enqueue([blockY, blockX, y, resX, pUncompressedData, pImage, &ILqt] {

      int16_t block[64 * 3];
      int16_t temp[64];

      for (size_t x = 0; x < blockX; x++)
      {
        swapFormatMCUBlock(block, pImage + ((y * blockX + x) << 6), 8, 8, (int)resX - 8);
        slapDCT((int16_t *)(pUncompressedData + (y * blockY + x) * DCT_PER_BLOCK_SIZE), temp, ILqt);
      }
    });
  }

  pQueue->wait();

  return result;
}

swapResult swapDecodeFrameYUV420(IN uint8_t * pUncompressedData, OUT uint8_t * pImage, const size_t resX, const size_t resY, mango::ConcurrentQueue * pQueue)
{
  swapResult result = sR_Success;

  const size_t blockX = resX >> 3;
  const size_t blockY = resY >> 3;

  uint8_t Lqt[64];
  uint8_t Cqt[64];
  uint16_t ILqt[64];
  uint16_t ICqt[64];
  uint32_t quality = 75;

  swapInitDctQuantizationTables(quality, Lqt, Cqt, ILqt, ICqt);
  
  for (size_t y = 0; y < blockY; y++)
  {
    pQueue->enqueue([&, blockY, blockX, y, resX, pUncompressedData, pImage] {
      for (size_t x = 0; x < blockX; x++)
        idct_sse2(pImage + ((y * blockX + x) << 6), (int)resX, (int16_t *)(pUncompressedData + (y * blockY + x) * DCT_PER_BLOCK_SIZE), (uint16_t *)Lqt);
    });
  }

  pQueue->wait();

  return result;
}
