#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
static void check(bool ok,const char*msg){if(!ok){std::cerr<<msg<<"\n";exit(1);}}
int main(int argc,char**argv){@autoreleasepool{
 check(argc==3||argc==4,"visibility_gpu source-directory output-directory [quality]");std::string out=argv[2];NSError*error=nil;
 auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];
 auto path=[@(argv[1]) stringByAppendingPathComponent:@"visibility.metal"];
 auto source=[NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
 NSMutableArray*pipes=[NSMutableArray array];NSMutableArray*extra=[NSMutableArray array];
 for(int fast=0;fast<2;fast++){
  auto opts=[MTLCompileOptions new];opts.fastMathEnabled=fast;
  auto lib=[device newLibraryWithSource:source options:opts error:&error];if(error)std::cerr<<[[error description] UTF8String]<<"\n";check(lib!=nil,"Shader compilation failed");
  auto p=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"visibility"] error:&error];check(p!=nil,"Pipeline failed");[pipes addObject:p];if(fast)for(NSString* name in @[@"classify",@"dispatch_args",@"queued_visibility"]){auto q=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&error];check(q!=nil,"Sparse pipeline failed");[extra addObject:q];}
 }
 std::ifstream input(out+"/visibility.bin",std::ios::binary);uint32_t dims[3];input.read((char*)dims,12);check(input.good()&&dims[0]==32&&dims[1]==32,"Invalid dataset");
 std::ofstream readbacks[3];readbacks[0].open(out+"/visibility_strict.bin",std::ios::binary);readbacks[1].open(out+"/visibility_fast.bin",std::ios::binary);
 readbacks[2].open(out+"/visibility_sparse.bin",std::ios::binary);
 std::ofstream report(out+"/visibility_timing.json");report<<"{\"device\":\""<<[[device name] UTF8String]<<"\",\"records\":[";bool first=true;
 for(uint32_t scene=0;scene<dims[2];scene++){
  uint32_t count;std::vector<float> primitives(80);input.read((char*)&count,4);input.read((char*)primitives.data(),320);check(input.good()&&count<=4,"Tile overflow/truncated source");
  for(auto shape:{std::pair<int,int>{32,32},{1920,1080}}){
   if(argc==4&&shape.first!=32)continue;
   int w=shape.first,h=shape.second,tx=(w+31)/32,ty=(h+31)/32,tiles=tx*ty;
   std::vector<uint32_t>counts(tiles,count);std::vector<float>all(tiles*80);for(int i=0;i<tiles;i++)std::copy(primitives.begin(),primitives.end(),all.begin()+i*80);
   auto pb=[device newBufferWithBytes:all.data() length:all.size()*4 options:MTLResourceStorageModeShared];auto cb=[device newBufferWithBytes:counts.data() length:counts.size()*4 options:MTLResourceStorageModeShared];
   auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:w height:h mipmapped:NO];d.storageMode=MTLStorageModeShared;d.usage=MTLTextureUsageShaderWrite;auto target=[device newTextureWithDescriptor:d];uint32_t info[4]={(uint32_t)w,(uint32_t)h,(uint32_t)tx,0};
   auto counter=[device newBufferWithLength:4 options:MTLResourceStorageModeShared];auto jobs=[device newBufferWithLength:w*h*4 options:MTLResourceStorageModePrivate];auto args=[device newBufferWithLength:12 options:MTLResourceStorageModePrivate];
   auto run=[&](int mode){auto cmd=[queue commandBuffer];if(mode==2){auto blit=[cmd blitCommandEncoder];[blit fillBuffer:counter range:NSMakeRange(0,4) value:0];[blit endEncoding];}
    for(int stage=0;stage<(mode==2?3:1);stage++){auto enc=[cmd computeCommandEncoder];[enc setComputePipelineState:mode==2?extra[stage]:pipes[mode]];[enc setBuffer:pb offset:0 atIndex:0];[enc setBuffer:cb offset:0 atIndex:1];[enc setBytes:info length:16 atIndex:2];[enc setTexture:target atIndex:0];[enc setBuffer:counter offset:0 atIndex:3];[enc setBuffer:jobs offset:0 atIndex:4];[enc setBuffer:args offset:0 atIndex:5];
     if(mode==2&&stage==1)[enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];else if(mode==2&&stage==2)[enc dispatchThreadgroupsWithIndirectBuffer:args indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(64,1,1)];else [enc dispatchThreads:MTLSizeMake(w,h,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];[enc endEncoding];}
    [cmd commit];[cmd waitUntilCompleted];check(cmd.status==MTLCommandBufferStatusCompleted,"GPU failed");return (cmd.GPUEndTime-cmd.GPUStartTime)*1000;};
   for(int mode=0;mode<3;mode++){
    run(mode);
    if(w==32){std::vector<float> pixels(w*h*4);[target getBytes:pixels.data() bytesPerRow:w*16 fromRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0];readbacks[mode].write((char*)pixels.data(),pixels.size()*4);}
   }
   std::vector<double>timings[3];for(int rep=0;rep<7;rep++)for(int m=0;m<3;m++){int mode=(m+rep)%3;timings[mode].push_back(run(mode));}
   for(int mode=0;mode<3;mode++){std::sort(timings[mode].begin(),timings[mode].end());if(!first)report<<",";first=false;report<<"{\"scene\":"<<scene<<",\"primitives_per_tile\":"<<count<<",\"width\":"<<w<<",\"height\":"<<h<<",\"mode\":\""<<(mode==2?"sparse":mode==1?"fast":"strict")<<"\",\"queued_fraction\":"<<(mode==2?double(*(uint32_t*)counter.contents)/(w*h):0)<<",\"fast_math\":"<<(mode?"true":"false")<<",\"gpu_ms\":"<<timings[mode][3]<<",\"p95_ms\":"<<timings[mode][6]<<"}";}
  }
  std::cerr<<"Visibility scene "<<scene<<" done\n";
 }
 report<<"]}\n";
}}
