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

#include "mango/core/thread.hpp"

using namespace swapcodec;

//////////////////////////////////////////////////////////////////////////

#define DCT_PER_BLOCK_SIZE 2048

//////////////////////////////////////////////////////////////////////////

void swapMemcpy(OUT void * pDestination, IN const void * pSource, const size_t size)
{
  apex_memcpy(pDestination, pSource, size);
}

void swapMemmove(OUT void * pDestination, IN_OUT void * pSource, const size_t size)
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

  pEncoder->pCompressibleData = new uint8_t[(resX >> 3) * (resY >> 3) * DCT_PER_BLOCK_SIZE];

  if (pEncoder->pCompressedData == nullptr)
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
    delete[]pCompressibleData;

  if (pThreadPool)
    delete (mango::ConcurrentQueue *)pThreadPool;
}

swapResult swapcodec::swapEncoder::AddFrameYUV420(IN_OUT uint8_t *pFrameData)
{
  swapResult result = sR_Success;

  if (sR_Success != (result = swapEncodeFrameYUV420(pFrameData, pCompressibleData, resX, resY, (mango::ConcurrentQueue *)pThreadPool)))
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

    pDestination[zigzag_table[i + 0 * 8]] = int16_t((v0 * pQuantizationTable[i + 0 * 8] + 0x4000) >> 15);
    pDestination[zigzag_table[i + 1 * 8]] = int16_t((v1 * pQuantizationTable[i + 1 * 8] + 0x4000) >> 15);
    pDestination[zigzag_table[i + 2 * 8]] = int16_t((v2 * pQuantizationTable[i + 2 * 8] + 0x4000) >> 15);
    pDestination[zigzag_table[i + 3 * 8]] = int16_t((v3 * pQuantizationTable[i + 3 * 8] + 0x4000) >> 15);
    pDestination[zigzag_table[i + 4 * 8]] = int16_t((v4 * pQuantizationTable[i + 4 * 8] + 0x4000) >> 15);
    pDestination[zigzag_table[i + 5 * 8]] = int16_t((v5 * pQuantizationTable[i + 5 * 8] + 0x4000) >> 15);
    pDestination[zigzag_table[i + 6 * 8]] = int16_t((v6 * pQuantizationTable[i + 6 * 8] + 0x4000) >> 15);
    pDestination[zigzag_table[i + 7 * 8]] = int16_t((v7 * pQuantizationTable[i + 7 * 8] + 0x4000) >> 15);
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
        swapFormatMCUBlock(block, pImage, 8, 8, (int)resX - 8);
        slapDCT((int16_t *)(pUncompressedData + (y * blockY + x) * DCT_PER_BLOCK_SIZE), temp, ILqt);
      }
    });
  }

  pQueue->wait();

  return result;
}
