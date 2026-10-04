#pragma once
#import <AudioUnit/AudioUnit.h>
#import <CoreAudio/CoreAudio.h>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>
#include "VSTHost.hpp"
inline void check(OSStatus s,const char* action) { if(s)throw std::runtime_error(std::string(action)+" ("+std::to_string(s)+")"); }
template<class T> T property(AudioObjectID id,AudioObjectPropertySelector key,AudioObjectPropertyScope scope=kAudioObjectPropertyScopeGlobal,AudioObjectPropertyElement element=kAudioObjectPropertyElementMain) {
    T value{}; UInt32 size=sizeof(value); AudioObjectPropertyAddress a{key,scope,element};
    check(AudioObjectGetPropertyData(id,&a,0,nullptr,&size,&value),"Read audio device property"); return value;
}
template<class T> void setProperty(AudioObjectID id,AudioObjectPropertySelector key,T value,AudioObjectPropertyScope scope=kAudioObjectPropertyScopeGlobal) {
    AudioObjectPropertyAddress a{key,scope,kAudioObjectPropertyElementMain}; check(AudioObjectSetPropertyData(id,&a,0,nullptr,sizeof(value),&value),"Set audio device property");
}
inline std::string deviceName(AudioObjectID id) { auto s=property<CFStringRef>(id,kAudioObjectPropertyName); char str[512]={}; CFStringGetCString(s,str,sizeof(str),kCFStringEncodingUTF8);CFRelease(s);return str; }
inline std::vector<AudioDeviceID> devices() {
    AudioObjectPropertyAddress a{kAudioHardwarePropertyDevices,kAudioObjectPropertyScopeGlobal,0};UInt32 size=0;
    check(AudioObjectGetPropertyDataSize(kAudioObjectSystemObject,&a,0,nullptr,&size),"List devices");std::vector<AudioDeviceID> ids(size/sizeof(AudioDeviceID));
    check(AudioObjectGetPropertyData(kAudioObjectSystemObject,&a,0,nullptr,&size,ids.data()),"List devices");return ids;
}
inline AudioDeviceID cableDevice() {for(auto id:devices())if(deviceName(id)=="VB-Cable")return id;throw std::runtime_error("VB-Cable is not installed or available");}
inline AudioDeviceID defaultOutput() {return property<AudioDeviceID>(kAudioObjectSystemObject,kAudioHardwarePropertyDefaultOutputDevice);}
inline AudioDeviceID systemOutput() {return property<AudioDeviceID>(kAudioObjectSystemObject,kAudioHardwarePropertyDefaultSystemOutputDevice);}
inline void route(AudioDeviceID id) {setProperty(kAudioObjectSystemObject,kAudioHardwarePropertyDefaultOutputDevice,id);}
inline bool alive(AudioDeviceID id) {try{return property<UInt32>(id,kAudioDevicePropertyDeviceIsAlive)!=0;}catch(...){return false;}}
inline void restore(AudioDeviceID cable,AudioDeviceID speaker,AudioDeviceID system) {
    try {if(defaultOutput()==cable && alive(speaker))route(speaker);}catch(...){}
    try {if(systemOutput()==cable && alive(system))setProperty(kAudioObjectSystemObject,kAudioHardwarePropertyDefaultSystemOutputDevice,system);}catch(...){}
}
struct StereoBuffers { UInt32 count=2; AudioBuffer buffers[2]; AudioBufferList* list(){return reinterpret_cast<AudioBufferList*>(this);} };
class AudioEngine {
    static constexpr uint64_t capacity=65536;
    static constexpr unsigned maxFrames=8192;
    float ring[2][capacity]{};
    float capture[2][maxFrames]{};
    float work[2][maxFrames]{}, result[2][maxFrames]{};
    std::atomic<uint64_t> writeIndex{0},readIndex{0};
    double position=0, correction=0;
    bool primed=false;
    AudioUnit input=nullptr,output=nullptr;
    std::unique_ptr<GMax> plugin;
    float lastGain=-1;
    static OSStatus inputCallback(void* context,AudioUnitRenderActionFlags* flags,const AudioTimeStamp* time,UInt32,UInt32 n,AudioBufferList*) {
        auto& e=*(AudioEngine*)context;
        if(n>maxFrames)return kAudio_ParamError;
        StereoBuffers b; for(int c=0;c<2;c++)b.buffers[c]={1,UInt32(n*sizeof(float)),e.capture[c]};
        OSStatus s=AudioUnitRender(e.input,flags,time,1,n,b.list()); if(s) {e.lastError=s;return s;}
        uint64_t w=e.writeIndex.load(std::memory_order_relaxed),r=e.readIndex.load(std::memory_order_acquire);
        if(w-r+n>=capacity) {e.overruns++;return noErr;}
        for(unsigned i=0;i<n;i++)for(int c=0;c<2;c++)e.ring[c][(w+i)%capacity]=e.capture[c][i];
        e.writeIndex.store(w+n,std::memory_order_release);e.captured.fetch_add(n,std::memory_order_relaxed);return noErr;
    }
    static OSStatus outputCallback(void* context,AudioUnitRenderActionFlags*,const AudioTimeStamp*,UInt32,UInt32 n,AudioBufferList* data) {
        auto& e=*(AudioEngine*)context;
        for(UInt32 b=0;b<data->mNumberBuffers;b++)memset(data->mBuffers[b].mData,0,data->mBuffers[b].mDataByteSize);
        if(n>maxFrames || data->mNumberBuffers<2)return noErr;
        uint64_t w=e.writeIndex.load(std::memory_order_acquire);
        if(!e.primed) {if(w-e.readIndex.load()<e.targetFrames+n)return noErr;e.position=(double)(w-e.targetFrames);e.primed=true;}
        double available=(double)w-e.position;
        if(available<n*1.002+2) {e.underruns++;e.primed=false;return noErr;}
        // Small continuous clock-drift correction; no block duplication or periodic clicks.
        double desired=std::clamp((available-e.targetFrames)/e.sampleRate*.01,-.001,.001);
        e.correction=.995*e.correction+.005*desired;
        double step=1+e.correction;
        float inPeak=0,outPeak=0;
        for(unsigned i=0;i<n;i++) {
            uint64_t r=(uint64_t)e.position;float f=(float)(e.position-r);
            for(int c=0;c<2;c++){float a=e.ring[c][r%capacity],b=e.ring[c][(r+1)%capacity];e.work[c][i]=a+(b-a)*f;inPeak=std::max(inPeak,std::abs(e.work[c][i]));}
            e.position+=step;
        }
        e.readIndex.store((uint64_t)e.position,std::memory_order_release);
        float gain=e.gainDB.load(); if(gain!=e.lastGain){e.plugin->fx->setParameter(e.plugin->fx,0,gain/24.f);e.lastGain=gain;}
        float* ins[2]={e.work[0],e.work[1]};float* outs[2]={e.result[0],e.result[1]};
        e.plugin->fx->processReplacing(e.plugin->fx,ins,outs,n);
        float ceiling=powf(10,-.3f/20);
        for(unsigned i=0;i<n;i++)for(int c=0;c<2;c++) {
            float v=e.result[c][i];if(!std::isfinite(v))v=0;
            // Final sample guard is defensive; GMax performs the actual limiting.
            v=std::clamp(v,-ceiling,ceiling);((float*)data->mBuffers[c].mData)[i]=v;outPeak=std::max(outPeak,std::abs(v));
        }
        e.inputPeak.store(inPeak);e.outputPeak.store(outPeak);e.rendered.fetch_add(n,std::memory_order_relaxed);return noErr;
    }
    AudioUnit makeUnit(AudioDeviceID device,bool isInput) {
        AudioComponentDescription d{kAudioUnitType_Output,kAudioUnitSubType_HALOutput,kAudioUnitManufacturer_Apple,0,0};
        AudioUnit unit=nullptr;check(AudioComponentInstanceNew(AudioComponentFindNext(nullptr,&d),&unit),"Create HAL audio unit");
        try {
            UInt32 one=1,zero=0;
            if(isInput){check(AudioUnitSetProperty(unit,kAudioOutputUnitProperty_EnableIO,kAudioUnitScope_Input,1,&one,sizeof(one)),"Enable cable input");check(AudioUnitSetProperty(unit,kAudioOutputUnitProperty_EnableIO,kAudioUnitScope_Output,0,&zero,sizeof(zero)),"Disable input-unit output");}
            check(AudioUnitSetProperty(unit,kAudioOutputUnitProperty_CurrentDevice,kAudioUnitScope_Global,0,&device,sizeof(device)),"Choose audio device");
            UInt32 max=maxFrames;check(AudioUnitSetProperty(unit,kAudioUnitProperty_MaximumFramesPerSlice,kAudioUnitScope_Global,0,&max,sizeof(max)),"Set callback capacity");
            AudioStreamBasicDescription format{};format.mSampleRate=sampleRate;format.mFormatID=kAudioFormatLinearPCM;
            format.mFormatFlags=kAudioFormatFlagsNativeFloatPacked|kAudioFormatFlagIsNonInterleaved;format.mBytesPerPacket=4;format.mFramesPerPacket=1;format.mBytesPerFrame=4;format.mChannelsPerFrame=2;format.mBitsPerChannel=32;
            check(AudioUnitSetProperty(unit,kAudioUnitProperty_StreamFormat,isInput?kAudioUnitScope_Output:kAudioUnitScope_Input,isInput?1:0,&format,sizeof(format)),"Set stereo format");
            AURenderCallbackStruct callback{isInput?inputCallback:outputCallback,this};
            check(AudioUnitSetProperty(unit,isInput?kAudioOutputUnitProperty_SetInputCallback:kAudioUnitProperty_SetRenderCallback,isInput?kAudioUnitScope_Global:kAudioUnitScope_Input,0,&callback,sizeof(callback)),"Install audio callback");
            check(AudioUnitInitialize(unit),"Initialize audio unit");return unit;
        } catch(...){AudioComponentInstanceDispose(unit);throw;}
    }
public:
    AudioDeviceID cable=0,speaker=0,previousSystem=0;
    double sampleRate=48000;
    uint64_t targetFrames=1536;
    std::atomic<float> gainDB{12},inputPeak{0},outputPeak{0};
    std::atomic<uint64_t> captured{0},rendered{0},underruns{0},overruns{0};
    std::atomic<int> lastError{0};
    AudioEngine(const char* pluginPath,float gain) {
        cable=cableDevice();speaker=defaultOutput();previousSystem=systemOutput();
        if(speaker==cable)throw std::runtime_error("Choose your speakers in macOS Sound, then enable normalization again");
        sampleRate=property<Float64>(speaker,kAudioDevicePropertyNominalSampleRate);
        if(sampleRate<32000 || sampleRate>192000)throw std::runtime_error("Unsupported speaker sample rate");
        setProperty(cable,kAudioDevicePropertyNominalSampleRate,sampleRate);
        targetFrames=(uint64_t)(sampleRate*.032);
        plugin=std::make_unique<GMax>(pluginPath,sampleRate);
        plugin->fx->setParameter(plugin->fx,0,gain/24.f);
        plugin->fx->setParameter(plugin->fx,1,.95f); // -0.3 dB ceiling
        plugin->fx->setParameter(plugin->fx,2,(.5f-.05f)/1.95f); // 500 ms release
        plugin->start();gainDB=gain;
        try {input=makeUnit(cable,true);output=makeUnit(speaker,false);check(AudioOutputUnitStart(input),"Start cable capture");check(AudioOutputUnitStart(output),"Start speaker playback");}
        catch(...){stop();throw;}
    }
    void enableRoute() {route(cable);}
    void stop() {
        restore(cable,speaker,previousSystem);
        if(output){AudioOutputUnitStop(output);AudioUnitUninitialize(output);AudioComponentInstanceDispose(output);output=nullptr;}
        if(input){AudioOutputUnitStop(input);AudioUnitUninitialize(input);AudioComponentInstanceDispose(input);input=nullptr;}
    }
    ~AudioEngine(){stop();}
};
