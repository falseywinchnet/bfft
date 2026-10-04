#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

static void check(bool ok,const char* message) { if(!ok) {std::cerr<<message<<"\n";exit(1);} }
int main(int argc,char**argv) { @autoreleasepool {
    check(argc>=4,"gpu shader.metal input.bin output-prefix [benchmark]");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();check(device!=nil,"No Metal device");
    NSError* error=nil;
    NSString* path=@(argv[1]);
    NSString* shader=[NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
    NSString* header=[NSString stringWithContentsOfFile:[[path stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"vendor/Fxaa3_11.h"] encoding:NSUTF8StringEncoding error:&error];
    check(shader!=nil && header!=nil,"Cannot read shader/header");
    shader=[shader stringByReplacingOccurrencesOfString:@"#include \"vendor/Fxaa3_11.h\"" withString:header];
    MTLCompileOptions* options=[MTLCompileOptions new]; options.fastMathEnabled=NO;
    id<MTLLibrary> library=[device newLibraryWithSource:shader options:options error:&error];
    if(error) std::cerr<<[[error description] UTF8String]<<"\n";
    check(library!=nil,"Cannot load library");
    id<MTLCommandQueue> queue=[device newCommandQueue];
    const char* names[]={"copyImage","fxaa12","tangentBox","tangentQuintic","tangentTensor"};
    constexpr int methods=5;
    NSMutableArray* pipes=[NSMutableArray array];
    for(auto name:names) {
        id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@(name)] error:&error];
        check(p!=nil,"Cannot make pipeline");[pipes addObject:p];
    }
    auto texture=[&](int w,int h, MTLPixelFormat format=MTLPixelFormatRGBA32Float) {
        auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
        d.storageMode=MTLStorageModeShared;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;
        return [device newTextureWithDescriptor:d];
    };
    auto run=[&](id<MTLTexture> src,id<MTLTexture> dst,int method,int repeats) {
        auto cb=[queue commandBuffer];
        auto encoder=[cb computeCommandEncoder];[encoder setComputePipelineState:pipes[method]];
        [encoder setTexture:src atIndex:0];[encoder setTexture:dst atIndex:1];
        for(int i=0;i<repeats;i++) {
            [encoder dispatchThreads:MTLSizeMake(src.width,src.height,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
            if(i+1<repeats) [encoder memoryBarrierWithScope:MTLBarrierScopeTextures];
        }
        [encoder endEncoding];[cb commit];[cb waitUntilCompleted];
        check(cb.status==MTLCommandBufferStatusCompleted,"GPU execution failed");
        return (cb.GPUEndTime-cb.GPUStartTime)*1000/repeats;
    };
    std::ifstream input(argv[2],std::ios::binary);uint32_t dims[3];input.read((char*)dims,12);
    check(input.good(),"Cannot read dataset");int w=dims[0],h=dims[1],count=dims[2];
    auto src=texture(w,h),dst=texture(w,h);std::vector<float> pixels(w*h*4),out(w*h*4);
    std::ofstream outputs[methods];for(int m=0;m<methods;m++) outputs[m].open(std::string(argv[3])+"_"+names[m]+".bin",std::ios::binary);
    for(int c=0;c<count;c++) {
        input.read((char*)pixels.data(),pixels.size()*4);check(input.good(),"Truncated dataset");
        [src replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:pixels.data() bytesPerRow:w*16];
        for(int m=0;m<methods;m++) {
            run(src,dst,m,1);
            [dst getBytes:out.data() bytesPerRow:w*16 fromRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0];
            outputs[m].write((char*)out.data(),out.size()*4);
        }
    }
    if(argc<5) return 0;
    std::ofstream timing(std::string(argv[3])+"_timing.json");
    timing<<"{\"device\":\""<<[[device name] UTF8String]<<"\",\"format\":\"RGBA8Unorm\",\"repeats_per_command\":8,\"records\":[";
    bool first=true;
    for(auto size: {std::pair<int,int>{1920,1080},{3840,2160}}) for(int scene=0;scene<3;scene++) {
        w=size.first;h=size.second;src=texture(w,h,MTLPixelFormatRGBA8Unorm);dst=texture(w,h,MTLPixelFormatRGBA8Unorm);pixels.resize(w*h*4);
        uint32_t rng=23;
        for(int y=0;y<h;y++) for(int x=0;x<w;x++) {
            rng=1664525*rng+1013904223;
            float v=scene==0 ? .4f : scene==1 ? float(std::sin(x*.03+y*.019)>0) : float(rng>>8)/16777215.0f;
            for(int ch=0;ch<4;ch++) pixels[(y*w+x)*4+ch]=v;
        }
        std::vector<uint8_t> encoded(pixels.size());
        for(size_t i=0;i<pixels.size();i++)encoded[i]=uint8_t(std::round(pixels[i]*255));
        [src replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:encoded.data() bytesPerRow:w*4];
        for(int m=0;m<methods;m++) run(src,dst,m,4);
        std::vector<double> times[methods],wall[methods];
        for(int rep=0;rep<21;rep++) for(int k=0;k<methods;k++) {
            int m=(k+rep)%methods;auto start=std::chrono::steady_clock::now();
            times[m].push_back(run(src,dst,m,8));
            wall[m].push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/8);
        }
        for(int m=0;m<methods;m++) {
            std::sort(times[m].begin(),times[m].end());std::sort(wall[m].begin(),wall[m].end());
            if(!first) timing<<",";first=false;
            timing<<"{\"width\":"<<w<<",\"height\":"<<h<<",\"scene\":"<<scene<<",\"method\":\""<<names[m]<<"\",\"gpu_ms_median\":"<<times[m][10]<<",\"gpu_ms_p95\":"<<times[m][19]<<",\"amortized_wall_ms\":"<<wall[m][10]<<",\"sorted_gpu_ms\":[";
            for(int i=0;i<21;i++){if(i)timing<<",";timing<<times[m][i];}timing<<"]}";
        }
        std::cerr<<"Timed "<<w<<"x"<<h<<" scene "<<scene<<"\n";
    }
    timing<<"]}\n";
} }
