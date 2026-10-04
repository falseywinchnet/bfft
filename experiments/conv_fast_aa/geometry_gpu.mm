#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
static void check(bool ok,const char* msg){if(!ok){std::cerr<<msg<<"\n";exit(1);}}
int main(int argc,char**argv){@autoreleasepool{
    check(argc==3 || argc==4,"geometry_gpu source-directory output-directory [fast]");
    NSString* root=@(argv[1]);std::string out=argv[2];
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();check(device!=nil,"Metal unavailable");
    NSError* error=nil;
    auto read=[&](NSString* file){return [NSString stringWithContentsOfFile:[root stringByAppendingPathComponent:file] encoding:NSUTF8StringEncoding error:&error];};
    NSString* shader=[read(@"aa.metal") stringByReplacingOccurrencesOfString:@"#include \"vendor/Fxaa3_11.h\"" withString:read(@"vendor/Fxaa3_11.h")];
    shader=[shader stringByAppendingString:read(@"geometry.metal")];
    bool halfCDF=argc==4 && std::string(argv[3])=="half";
    if(halfCDF){shader=[shader stringByReplacingOccurrencesOfString:@"fragment float4 areaFragment" withString:@"float edgeCoverageHalf(float,float2);\nfragment float4 areaFragment"];shader=[shader stringByReplacingOccurrencesOfString:@"area=edgeCoverage(d,normal)" withString:@"area=edgeCoverageHalf(d,normal)"];shader=[shader stringByAppendingString:read(@"half_coverage.metal")];}
    MTLCompileOptions* options=[MTLCompileOptions new];options.fastMathEnabled=argc==4;
    id<MTLLibrary> library=[device newLibraryWithSource:shader options:options error:&error];
    if(error)std::cerr<<[[error description] UTF8String]<<"\n";check(library!=nil,"Shader compile failed");
    id<MTLCommandQueue> queue=[device newCommandQueue];
    auto renderPipeline=[&](bool analytic,int samples,bool additive=false){
        auto d=[MTLRenderPipelineDescriptor new];d.vertexFunction=[library newFunctionWithName:analytic?@"boundsVertex":@"triangleVertex"];
        d.fragmentFunction=[library newFunctionWithName:analytic?@"areaFragment":@"opaqueFragment"];d.rasterSampleCount=samples;
        d.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA8Unorm;
        if(analytic){d.colorAttachments[0].blendingEnabled=YES;d.colorAttachments[0].sourceRGBBlendFactor=MTLBlendFactorSourceAlpha;d.colorAttachments[0].destinationRGBBlendFactor=additive?MTLBlendFactorOne:MTLBlendFactorOneMinusSourceAlpha;d.colorAttachments[0].sourceAlphaBlendFactor=MTLBlendFactorOne;d.colorAttachments[0].destinationAlphaBlendFactor=additive?MTLBlendFactorOne:MTLBlendFactorZero;}
        auto p=[device newRenderPipelineStateWithDescriptor:d error:&error];check(p!=nil,"Render pipeline failed");return p;
    };
    auto point=renderPipeline(false,1),msaa=renderPipeline(false,4),area=renderPipeline(true,1);
    auto sumArea=renderPipeline(true,1,true);
    auto fxaa=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"fxaa12"] error:&error];check(fxaa!=nil,"FXAA pipeline failed");
    std::ofstream report(out+"/geometry_timing.json");report<<"{\"fast_math\":"<<(argc==4?"true":"false")<<",\"device\":\""<<[[device name] UTF8String]<<"\",\"sample_positions\":[";
    MTLSamplePosition samplePositions[4];[device getDefaultSamplePositions:samplePositions count:4];
    for(int i=0;i<4;i++){if(i)report<<",";report<<"["<<samplePositions[i].x<<","<<samplePositions[i].y<<"]";}report<<"],\"records\":[";
    bool first=true;
    for(auto size:{std::pair<int,int>{512,512},{1920,1080},{3840,2160}})for(int scene=0;scene<(size.first==512?6:2);scene++){
        int w=size.first,h=size.second;float dims[2]={(float)w,(float)h};std::vector<float> vertices;
        // Fixed separated tiles: no hidden visibility/order problem in this diagnostic.
        const int spacing=scene==0?64:scene==1?16:32;
        for(int y=spacing/2;y<h-spacing/2;y+=spacing)for(int x=spacing/2;x<w-spacing/2;x+=spacing){
            if(scene==4){float r=spacing*.3f;float square[12]={x-r,y-r,x+r,y-r,x+r,y+r,x-r,y-r,x+r,y+r,x-r,y+r};vertices.insert(vertices.end(),square,square+12);continue;}
            float angle=float((x*31+y*17)%997)*.013f,cs=cosf(angle),sn=sinf(angle);
            float v[6]={-.34f*spacing,-.26f*spacing,.34f*spacing,-.26f*spacing,0,.34f*spacing};
            if(scene==2){v[1]=-.04f;v[3]=-.04f;v[5]=.04f;}
            if(scene==3){cs=1;sn=0;v[0]=v[1]=v[3]=v[4]=-spacing*.25f;v[2]=v[5]=spacing*.25f;}
            for(int k=0;k<3;k++){vertices.push_back(x+(scene==3?0:.173f)+cs*v[2*k]-sn*v[2*k+1]);vertices.push_back(y+(scene==3?0:.291f)+sn*v[2*k]+cs*v[2*k+1]);}
            if(scene==5){std::vector<float> duplicate(vertices.end()-6,vertices.end());vertices.insert(vertices.end(),duplicate.begin(),duplicate.end());}
        }
        int count=vertices.size()/6;auto vb=[device newBufferWithBytes:vertices.data() length:vertices.size()*4 options:MTLResourceStorageModeShared];
        auto makeTexture=[&](int samples){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:w height:h mipmapped:NO];d.sampleCount=samples;
            d.textureType=samples>1?MTLTextureType2DMultisample:MTLTextureType2D;d.storageMode=samples>1?MTLStorageModePrivate:MTLStorageModeShared;d.usage=samples>1?MTLTextureUsageRenderTarget:MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite;return [device newTextureWithDescriptor:d];};
        auto single=makeTexture(1),multi=makeTexture(4),filtered=makeTexture(1);
        auto run=[&](int method,int repeats){
            auto cb=[queue commandBuffer];
            for(int rep=0;rep<repeats;rep++){
                auto pass=[MTLRenderPassDescriptor renderPassDescriptor];auto att=pass.colorAttachments[0];att.texture=method==2?multi:single;att.loadAction=MTLLoadActionClear;att.clearColor=MTLClearColorMake(0,0,0,0);
                att.storeAction=method==2?MTLStoreActionMultisampleResolve:MTLStoreActionStore;if(method==2)att.resolveTexture=single;
                auto enc=[cb renderCommandEncoderWithDescriptor:pass];[enc setRenderPipelineState:method==2?msaa:method==3?area:method==4?sumArea:point];[enc setVertexBuffer:vb offset:0 atIndex:0];[enc setVertexBytes:dims length:8 atIndex:1];[enc setFragmentBuffer:vb offset:0 atIndex:0];[enc setCullMode:MTLCullModeNone];
                [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:method>=3?6:3 instanceCount:count];[enc endEncoding];
                if(method==1){auto ce=[cb computeCommandEncoder];[ce setComputePipelineState:fxaa];[ce setTexture:single atIndex:0];[ce setTexture:filtered atIndex:1];[ce dispatchThreads:MTLSizeMake(w,h,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];[ce endEncoding];}
            }
            [cb commit];[cb waitUntilCompleted];check(cb.status==MTLCommandBufferStatusCompleted,"Render failed");return (cb.GPUEndTime-cb.GPUStartTime)*1000/repeats;
        };
        const char* names[]={"point","point_fxaa12","msaa4","analytic_boundary","boundary_sum"};
        constexpr int methods=5;
        if(w==512){std::ofstream vf(out+"/triangles_"+std::to_string(scene)+".bin",std::ios::binary);vf.write((char*)vertices.data(),vertices.size()*4);}
        for(int m=0;m<methods;m++){
            run(m,4);
            if(w==512){std::vector<uint8_t> pixels(w*h*4);auto tex=m==1?filtered:single;[tex getBytes:pixels.data() bytesPerRow:w*4 fromRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0];std::ofstream f(out+"/render_"+std::to_string(scene)+"_"+names[m]+".bin",std::ios::binary);f.write((char*)pixels.data(),pixels.size());}
        }
        std::vector<double> times[methods];for(int rep=0;rep<21;rep++)for(int k=0;k<methods;k++){int m=(rep+k)%methods;times[m].push_back(run(m,8));}
        for(int m=0;m<methods;m++){std::sort(times[m].begin(),times[m].end());if(!first)report<<",";first=false;report<<"{\"width\":"<<w<<",\"height\":"<<h<<",\"scene\":"<<scene<<",\"triangles\":"<<count<<",\"method\":\""<<names[m]<<"\",\"gpu_ms_median\":"<<times[m][10]<<",\"gpu_ms_p95\":"<<times[m][19]<<"}";}
        std::cerr<<"Geometry "<<w<<"x"<<h<<" scene "<<scene<<"\n";
    }
    report<<"]}\n";
}}
