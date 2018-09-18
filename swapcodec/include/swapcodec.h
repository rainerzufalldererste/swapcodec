// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef swapcodec_h__
#define swapcodec_h__

#include <stdint.h>
#include <string>

#ifndef IN
#define IN
#endif // !IN

#ifndef OUT
#define OUT
#endif // !OUT

#ifndef IN_OUT
#define IN_OUT IN OUT
#endif // !IN_OUT

#define SSSE3

namespace swapcodec
{
  enum swapResult
  {
    sR_Success,
    sR_Failure,
    sR_InternalError,
    sR_MemoryAllocationFailure
  };

  void swapMemcpy(OUT void *pDestination, IN const void *pSource, const size_t size);
  void swapMemmove(OUT void *pDestination, IN_OUT void *pSource, const size_t size);

  struct swapEncoder
  {
    static swapEncoder * Create(const std::string &filename, const size_t resX, const size_t resY);
    ~swapEncoder();

    swapResult AddFrameYUV420(IN_OUT uint8_t *pFrameData);
    swapResult Finalize();

    uint8_t *pLowResDataUncompressed = nullptr;
    uint8_t *pLastFrameUncompressed = nullptr;

    uint8_t *pCompressibleData = nullptr;
    uint8_t *pCompressedData = nullptr;
    size_t compressedDataCapacity = 0;
    size_t compressedDataSize = 0;

    size_t resX;
    size_t resY;
    size_t lowResX;
    size_t lowResY;
    size_t currentFrameIndex;
    size_t iframeStep;

    std::string filename;
    FILE *pHeaderFile = nullptr;
    FILE *pMainFile = nullptr;
    FILE *pFinalFile = nullptr;

    void *pThreadPool = nullptr;
  };

  struct swapDecoder
  {
    static swapDecoder * Create();
    ~swapDecoder();

    std::string filename;
    FILE *pFile;

    uint8_t *pFrameData = nullptr;
    size_t frameDataCapacity = 0;
    size_t frameDataSize = 0;

    size_t resX;
    size_t resY;
    size_t lowResX;
    size_t lowResY;
    size_t currentFrameIndex;
    size_t iframeStep;

    uint8_t *pDecodedFrameYUV420 = nullptr;
  };
}

#endif // swapcodec_h__
