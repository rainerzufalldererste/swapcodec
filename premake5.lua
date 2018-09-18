solution "swapcodec"
  
  editorintegration "On"
  configurations { "Debug", "Release" }
  platforms { "x64" }

  dofile "swapcodec/project.lua"
    location("swapcodec")

  dofile "TestApp/project.lua"
    location("TestApp")