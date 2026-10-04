#pragma once
#include <CoreFoundation/CoreFoundation.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
struct Effect;
using Dispatch = intptr_t (*)(Effect*, int32_t, int32_t, intptr_t, void*, float);
using Process = void (*)(Effect*, float**, float**, int32_t);
struct Effect {
    int32_t magic; Dispatch dispatcher; Process process;
    void (*setParameter)(Effect*, int32_t, float);
    float (*getParameter)(Effect*, int32_t);
    int32_t numPrograms, numParams, numInputs, numOutputs, flags;
    intptr_t reserved1, reserved2;
    int32_t initialDelay, realQualities, offQualities;
    float ioRatio;
    void *object, *user;
    int32_t uniqueID, version;
    Process processReplacing;
    void (*processDoubleReplacing)(Effect*, double**, double**, int32_t);
    char future[56];
};
class GMax {
    CFBundleRef bundle = nullptr;
public:
    Effect* fx = nullptr;
    static inline double rate = 48000;
    static inline int block = 8192;
    static intptr_t host(Effect*, int32_t op, int32_t, intptr_t, void* ptr, float) {
        switch(op) {
            case 1: return 2400; // audioMasterVersion
            case 16: return (intptr_t)rate;
            case 17: return block;
            case 32: if(ptr) snprintf((char*)ptr, 64, "Personal Audio"); return 1;
            case 33: if(ptr) snprintf((char*)ptr, 64, "TV Normalizer"); return 1;
            case 34: return 1000;
            case 37: return 1; // language: English
            default: return 0;
        }
    }
    GMax(const char* path, double sampleRate) {
        rate = sampleRate;
        auto url = CFURLCreateFromFileSystemRepresentation(nullptr, (const UInt8*)path, strlen(path), true);
        bundle = CFBundleCreate(nullptr, url); CFRelease(url);
        if(!bundle || !CFBundleLoadExecutable(bundle)) throw std::runtime_error("Cannot load the installed GMax.vst");
        using Entry = Effect* (*)(Dispatch);
        auto entry = (Entry)CFBundleGetFunctionPointerForName(bundle, CFSTR("VSTPluginMain"));
        if(!entry || !(fx=entry(host)) || fx->magic != 0x56737450 || !fx->processReplacing)
            throw std::runtime_error("GMax returned an unsupported VST interface");
        fx->dispatcher(fx,0,0,0,nullptr,0);
        fx->dispatcher(fx,10,0,0,nullptr,(float)rate);
        fx->dispatcher(fx,11,0,block,nullptr,0);
        if(fx->numInputs!=2 || fx->numOutputs!=2) throw std::runtime_error("GMax is not stereo");
    }
    void start() { fx->dispatcher(fx,12,0,1,nullptr,0); fx->dispatcher(fx,71,0,0,nullptr,0); }
    std::string text(int opcode,int index) { char s[256]={}; fx->dispatcher(fx,opcode,index,0,s,0); return s; }
    float parameterFor(int index,float desired) {
        fx->setParameter(fx,index,0); float a=std::stof(text(7,index));
        fx->setParameter(fx,index,1); float b=std::stof(text(7,index));
        float lo=0,hi=1;
        for(int i=0;i<24;i++) { float mid=(lo+hi)/2; fx->setParameter(fx,index,mid); float v=std::stof(text(7,index));
            if((v<desired)==(b>a)) lo=mid; else hi=mid;
        }
        return (lo+hi)/2;
    }
    ~GMax() { if(fx) {fx->dispatcher(fx,72,0,0,nullptr,0); fx->dispatcher(fx,12,0,0,nullptr,0);fx->dispatcher(fx,1,0,0,nullptr,0);} if(bundle)CFRelease(bundle); }
};
