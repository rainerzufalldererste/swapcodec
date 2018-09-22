// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "swapcodec.h"
#include <inttypes.h>

using namespace swapcodec;

int main(int argc, char **pArgv)
{
  void *pFileData = nullptr;
  void *pFrame = nullptr;
  int retval = 0;
  char *origFile = nullptr;
  char *slapFile = nullptr;

  if (argc > 2)
  {
    origFile = pArgv[1];
    slapFile = pArgv[2];

    FILE *pFile = fopen(origFile, "rb");

    if (!pFile)
    {
      printf("File not found.");
      retval = 1;
      goto epilogue;
    }

    fseek(pFile, 0, SEEK_END);
    size_t size = ftell(pFile);
    fseek(pFile, 0, SEEK_SET);

    pFileData = malloc(7680 * 11520);

    if (!pFileData)
    {
      printf("Memory allocation failure.");
      retval = 1;
      goto epilogue;
    }

    size_t readBytes = fread(pFileData, 1, size, pFile);
    printf("Read %" PRIu64 " bytes from '%s'.\n", readBytes, pArgv[1]);

    fclose(pFile);
  }
  else if (argc == 2)
  {
    slapFile = pArgv[1];
  }
  else
  {
    printf("Usage: %s <inputfile> <outputfile>", pArgv[0]);
    goto epilogue;
  }

  size_t frameCount;

  if (origFile)
  {
    frameCount = 100;
    printf("Adding %" PRIu64 " frames...\n", frameCount);

    pFrame = malloc(7680 * 11520);

    swapEncoder *pEncoder = swapEncoder::Create("", 7680, 7680);

    for (size_t i = 0; i < frameCount; i++)
    {
      swapMemcpy(pFrame, pFileData, 7680 * 11520);

      if (pEncoder->AddFrameYUV420((uint8_t *)pFrame))
        __debugbreak();

      if (i == 0)
      {
        FILE *pTestFile = fopen(slapFile, "wb");
        fwrite(pFrame, 1, 7680 * 11520, pTestFile);
        fclose(pTestFile);
      }

      printf("\rFrame %" PRIu64 " / %" PRIu64 " processed.", i + 1, frameCount);
    }
  }

epilogue:
  return retval;
}
