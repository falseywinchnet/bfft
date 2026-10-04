#import <Cocoa/Cocoa.h>
#import <AVFoundation/AVFoundation.h>
#include <unistd.h>
#include <signal.h>
#include <fstream>
#include "AudioEngine.hpp"
static NSString* supportPath() {return [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Application Support/TV Normalizer"];}
static NSString* pluginPath() {return [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Audio/Plug-Ins/VST/GMax.vst"];}
static NSString* recoveryPath(){return [supportPath() stringByAppendingPathComponent:@"recovery.json"];}
static void recoverLastRoute() {
    NSData* d=[NSData dataWithContentsOfFile:recoveryPath()];if(!d)return;
    NSDictionary* s=[NSJSONSerialization JSONObjectWithData:d options:0 error:nil];
    if(s)restore([s[@"cable"] unsignedIntValue],[s[@"speaker"] unsignedIntValue],[s[@"system"] unsignedIntValue]);
    [[NSFileManager defaultManager]removeItemAtPath:recoveryPath() error:nil];
}
@interface AppDelegate : NSObject <NSApplicationDelegate,NSMenuDelegate> {
    std::unique_ptr<AudioEngine> engine;
    NSStatusItem* status;
    NSMenuItem *toggleItem,*deviceItem,*meterItem,*loginItem;
    NSMenu* boostMenu;
    NSTimer* timer;
    NSTask* watchdog;
    NSString* lastMessage;
    float gain;
    BOOL busy,wakeEnabled;
    uint64_t lastCaptured,lastRendered;
    int stalled;
}
@end
@implementation AppDelegate
- (void)applicationDidFinishLaunching:(NSNotification*)note {
    [[NSFileManager defaultManager]createDirectoryAtPath:supportPath() withIntermediateDirectories:YES attributes:nil error:nil];
    recoverLastRoute();
    NSUserDefaults* prefs=NSUserDefaults.standardUserDefaults;
    [prefs registerDefaults:@{@"boostDB":@12}];gain=std::clamp([prefs floatForKey:@"boostDB"],0.f,24.f);
    status=[NSStatusBar.systemStatusBar statusItemWithLength:NSVariableStatusItemLength];
    status.button.title=@"GMax OFF";status.button.toolTip=@"TV Normalizer — click to enable or disable audio normalization";
    NSMenu* menu=[[NSMenu alloc]init];menu.delegate=self;
    toggleItem=[[NSMenuItem alloc]initWithTitle:@"Enable normalization" action:@selector(toggle:) keyEquivalent:@""];toggleItem.target=self;[menu addItem:toggleItem];
    [menu addItem:NSMenuItem.separatorItem];
    NSMenuItem* boost=[[NSMenuItem alloc]initWithTitle:@"Volume boost" action:nil keyEquivalent:@""];boostMenu=[[NSMenu alloc]init];
    for(int db:{0,3,6,9,12,15,18,21,24}) {NSMenuItem* item=[[NSMenuItem alloc]initWithTitle:[NSString stringWithFormat:@"+%d dB%@",db,db==12?@"  ·  Default":@""] action:@selector(setGain:) keyEquivalent:@""];item.tag=db;item.target=self;[boostMenu addItem:item];}boost.submenu=boostMenu;[menu addItem:boost];
    deviceItem=[[NSMenuItem alloc]initWithTitle:@"Output: speakers" action:nil keyEquivalent:@""];[menu addItem:deviceItem];
    meterItem=[[NSMenuItem alloc]initWithTitle:@"GMax limiter · −0.3 dB ceiling" action:nil keyEquivalent:@""];[menu addItem:meterItem];
    [menu addItem:NSMenuItem.separatorItem];
    loginItem=[[NSMenuItem alloc]initWithTitle:@"Launch at login" action:@selector(toggleLogin:) keyEquivalent:@""];loginItem.target=self;[menu addItem:loginItem];
    NSMenuItem* about=[[NSMenuItem alloc]initWithTitle:@"About TV Normalizer…" action:@selector(about:) keyEquivalent:@""];about.target=self;[menu addItem:about];
    NSMenuItem* quit=[[NSMenuItem alloc]initWithTitle:@"Quit and restore audio" action:@selector(quit:) keyEquivalent:@"q"];quit.target=self;[menu addItem:quit];status.menu=menu;
    [NSWorkspace.sharedWorkspace.notificationCenter addObserver:self selector:@selector(willSleep:) name:NSWorkspaceWillSleepNotification object:nil];
    [NSWorkspace.sharedWorkspace.notificationCenter addObserver:self selector:@selector(didWake:) name:NSWorkspaceDidWakeNotification object:nil];
    timer=[NSTimer scheduledTimerWithTimeInterval:1 target:self selector:@selector(tick:) userInfo:nil repeats:YES];
    [self enable];
}
- (NSString*)loginPath {return [NSHomeDirectory() stringByAppendingPathComponent:@"Library/LaunchAgents/local.personal.TVNormalizer.plist"];}
- (void)toggleLogin:(id)sender {
    NSString* path=[self loginPath];
    if([[NSFileManager defaultManager]fileExistsAtPath:path])[[NSFileManager defaultManager]removeItemAtPath:path error:nil];
    else {
        [[NSFileManager defaultManager]createDirectoryAtPath:path.stringByDeletingLastPathComponent withIntermediateDirectories:YES attributes:nil error:nil];
        NSDictionary* p=@{@"Label":@"local.personal.TVNormalizer",@"ProgramArguments":@[NSBundle.mainBundle.executablePath],@"RunAtLoad":@YES,@"ProcessType":@"Interactive",@"LimitLoadToSessionType":@"Aqua"};
        [p writeToFile:path atomically:YES];
    }
    [self refresh];
}
- (void)about:(id)sender {
    NSAlert* a=[[NSAlert alloc]init];a.messageText=@"TV Normalizer";
    a.informativeText=@"Runs the actual GVST GMax 3.1 VST plug-in locally.\n\nEnable: system audio → VB-Cable → GMax → your selected speakers.\nDisable or quit: restore direct speaker output.\n\nBoost: 0–24 dB · Peak ceiling: −0.3 dB · Release: 0.5 seconds. Audio is processed live and is never recorded or uploaded.\n\nGMax © Graham Yeadon · gvst.uk";[a runModal];
}
- (void)showError:(NSString*)message {
    lastMessage=message;NSAlert* alert=[[NSAlert alloc]init];alert.messageText=@"Normalization could not start";alert.informativeText=message;[alert runModal];
}
- (void)enable {
    if(engine||busy)return;busy=YES;lastMessage=nil;[self refresh];
    AVAuthorizationStatus auth=[AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio];
    if(auth==AVAuthorizationStatusNotDetermined) {
        [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio completionHandler:^(BOOL granted){dispatch_async(dispatch_get_main_queue(),^{self->busy=NO;if(granted)[self enable];else [self showError:@"macOS needs audio input permission to read VB-Cable. Allow TV Normalizer in System Settings → Privacy & Security → Microphone, then enable normalization."];});}];return;
    }
    if(auth!=AVAuthorizationStatusAuthorized){busy=NO;[self showError:@"Allow TV Normalizer in System Settings → Privacy & Security → Microphone so it can receive VB-Cable audio."];[self refresh];return;}
    try {
        engine=std::make_unique<AudioEngine>(pluginPath().fileSystemRepresentation,gain);
        NSDictionary* recovery=@{@"cable":@(engine->cable),@"speaker":@(engine->speaker),@"system":@(engine->previousSystem)};
        NSData* data=[NSJSONSerialization dataWithJSONObject:recovery options:0 error:nil];
        if(![data writeToFile:recoveryPath() atomically:YES])throw std::runtime_error("Could not save the audio recovery information");
        watchdog=[[NSTask alloc]init];watchdog.executableURL=NSBundle.mainBundle.executableURL;
        watchdog.arguments=@[@"--watch",[NSString stringWithFormat:@"%d",getpid()],[NSString stringWithFormat:@"%u",engine->cable],[NSString stringWithFormat:@"%u",engine->speaker],[NSString stringWithFormat:@"%u",engine->previousSystem]];
        NSError* error=nil;if(![watchdog launchAndReturnError:&error])throw std::runtime_error("Could not start the audio recovery watchdog");
        engine->enableRoute();lastCaptured=lastRendered=0;stalled=0;
    } catch(const std::exception& e){[self disable];[self showError:[NSString stringWithUTF8String:e.what()]];}
    busy=NO;[self refresh];
}
- (void)disable {
    engine.reset();
    if(watchdog.running)[watchdog terminate];watchdog=nil;
    [[NSFileManager defaultManager]removeItemAtPath:recoveryPath() error:nil];
    [self refresh];
}
- (void)toggle:(id)sender {if(engine)[self disable];else [self enable];}
- (void)setGain:(NSMenuItem*)sender {gain=sender.tag;[NSUserDefaults.standardUserDefaults setFloat:gain forKey:@"boostDB"];if(engine)engine->gainDB=gain;[self refresh];}
- (void)refresh {
    status.button.title=engine?@"GMax ON":@"GMax OFF";
    toggleItem.title=engine?@"Disable normalization":@"Enable normalization";toggleItem.state=engine?NSControlStateValueOn:NSControlStateValueOff;toggleItem.enabled=!busy;
    for(NSMenuItem* item in boostMenu.itemArray)item.state=item.tag==(int)gain?NSControlStateValueOn:NSControlStateValueOff;
    loginItem.state=[[NSFileManager defaultManager]fileExistsAtPath:[self loginPath]]?NSControlStateValueOn:NSControlStateValueOff;
    try {deviceItem.title=[NSString stringWithFormat:@"Output: %s",deviceName(engine?engine->speaker:defaultOutput()).c_str()];}catch(...){deviceItem.title=@"Output unavailable";}
    status.button.toolTip=lastMessage?:[NSString stringWithFormat:@"Normalization %@ · +%.0f dB · click for controls",engine?@"on":@"off",gain];
}
- (void)menuWillOpen:(NSMenu*)menu {[self refresh];}
- (void)tick:(id)sender {
    NSMutableDictionary* info=[@{@"enabled":@(bool(engine)),@"boostDB":@(gain),@"plugin":@"GVST GMax 3.1",@"message":lastMessage?:@"",@"pid":@(getpid()),@"menuBarVisible":@(status.visible)} mutableCopy];
    if(engine){
        uint64_t c=engine->captured.load(),r=engine->rendered.load();
        info[@"capturedFrames"]=@(c);info[@"renderedFrames"]=@(r);info[@"inputPeak"]=@(engine->inputPeak.load());info[@"outputPeak"]=@(engine->outputPeak.load());info[@"underruns"]=@(engine->underruns.load());info[@"overruns"]=@(engine->overruns.load());info[@"sampleRate"]=@(engine->sampleRate);
        if(c==lastCaptured||r==lastRendered)stalled++;else stalled=0;lastCaptured=c;lastRendered=r;
        try {
            info[@"outputDevice"]=[NSString stringWithUTF8String:deviceName(engine->speaker).c_str()];
            if(defaultOutput()!=engine->cable){lastMessage=@"Normalization stopped because the audio output changed.";[self disable];}
            else if(stalled>=4||!alive(engine->speaker)||engine->lastError.load()) {lastMessage=@"Audio routing stopped; direct speaker output restored. Click to enable again.";[self disable];}
            else if(property<Float64>(engine->speaker,kAudioDevicePropertyNominalSampleRate)!=engine->sampleRate || property<Float64>(engine->cable,kAudioDevicePropertyNominalSampleRate)!=engine->sampleRate) {lastMessage=@"Audio sample rate changed; direct output restored. Click to enable again.";[self disable];}
        }catch(...){lastMessage=@"Audio device became unavailable.";[self disable];}
    }
    info[@"enabled"]=@(bool(engine));info[@"message"]=lastMessage?:@"";
    NSData* json=[NSJSONSerialization dataWithJSONObject:info options:NSJSONWritingPrettyPrinted error:nil];[json writeToFile:[supportPath() stringByAppendingPathComponent:@"status.json"] atomically:YES];
}
- (void)willSleep:(id)n {wakeEnabled=bool(engine);[self disable];}
- (void)didWake:(id)n {if(wakeEnabled){wakeEnabled=NO;dispatch_after(dispatch_time(DISPATCH_TIME_NOW,2*NSEC_PER_SEC),dispatch_get_main_queue(),^{[self enable];});}}
- (void)quit:(id)sender {[NSApp terminate:nil];}
- (void)applicationWillTerminate:(NSNotification*)n {[self disable];}
@end
static int selfTest() {
    GMax p(pluginPath().fileSystemRepresentation,48000);
    p.fx->setParameter(p.fx,0,.5);p.fx->setParameter(p.fx,1,.95);p.fx->setParameter(p.fx,2,(.5f-.05f)/1.95f);p.start();
    constexpr int N=1024;float in[2][N]{},out[2][N]{};float* ip[2]={in[0],in[1]},*op[2]={out[0],out[1]};
    double sumIn=0,sumOut=0;float peak=0;
    for(int b=0;b<40;b++){for(int i=0;i<N;i++){float v=.01f*sinf(2*M_PI*440*(b*N+i)/48000);in[0][i]=in[1][i]=v;}p.fx->processReplacing(p.fx,ip,op,N);if(b>5)for(int i=0;i<N;i++){sumIn+=in[0][i]*in[0][i];sumOut+=out[0][i]*out[0][i];if(std::abs(out[0][i]-out[1][i])>1e-7)throw std::runtime_error("Stereo mismatch");}}
    double db=10*log10(sumOut/sumIn);if(std::abs(db-12)>.1)throw std::runtime_error("Quiet-signal gain test failed");
    for(int b=0;b<80;b++){for(int i=0;i<N;i++){in[0][i]=1.5f*sinf(2*M_PI*997*(b*N+i)/48000);in[1][i]=.8f*in[0][i];}p.fx->processReplacing(p.fx,ip,op,N);for(int i=0;i<N;i++)for(int c=0;c<2;c++){if(!std::isfinite(out[c][i]))throw std::runtime_error("Nonfinite output");peak=std::max(peak,std::abs(out[c][i]));}}
    if(peak>pow(10,-.3/20)+1e-5)throw std::runtime_error("Peak ceiling test failed");
    memset(in,0,sizeof(in));for(int b=0;b<200;b++)p.fx->processReplacing(p.fx,ip,op,N);
    float silence=0;for(int i=0;i<N;i++)silence=std::max(silence,std::abs(out[0][i]));if(silence>1e-10)throw std::runtime_error("Silence test failed");
    printf("PASS: actual GMax quiet gain %.4f dB; overload peak %.6f (ceiling %.6f); stereo matched; silence %.8g; plugin latency %d frames\n",db,peak,pow(10,-.3/20),silence,p.fx->initialDelay);return 0;
}
int main(int argc,char**argv) {@autoreleasepool {
    try {
        if(argc>1 && !strcmp(argv[1],"--watch") && argc==6){pid_t parent=atoi(argv[2]);while(kill(parent,0)==0)usleep(500000);restore(atoi(argv[3]),atoi(argv[4]),atoi(argv[5]));return 0;}
        if(argc>1 && !strcmp(argv[1],"--restore")){recoverLastRoute();return 0;}
        if(argc>1 && !strcmp(argv[1],"--devices")){for(auto d:devices())printf("%u %s%s\n",d,deviceName(d).c_str(),d==defaultOutput()?" [default output]":"");return 0;}
        [NSApplication sharedApplication];[NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
        if(argc>1 && !strcmp(argv[1],"--self-test"))return selfTest();
        if(argc>1 && !strcmp(argv[1],"--route-test")) {
            AudioDeviceID before=defaultOutput();
            auto e=std::make_unique<AudioEngine>(pluginPath().fileSystemRepresentation,12);
            e->enableRoute();if(defaultOutput()!=e->cable)throw std::runtime_error("Route enable failed");
            usleep(500000);
            if(e->captured.load()==0 || e->rendered.load()==0)throw std::runtime_error("Audio callbacks did not run");
            printf("PASS: route enabled, captured %llu / rendered %llu frames",(unsigned long long)e->captured.load(),(unsigned long long)e->rendered.load());
            e.reset();if(defaultOutput()!=before)throw std::runtime_error("Route restore failed");printf("; direct output restored\n");return 0;
        }
        // A second launch must not create a second audio host.
        if([NSRunningApplication runningApplicationsWithBundleIdentifier:@"local.personal.TVNormalizer"].count>1)return 0;
        AppDelegate* delegate=[[AppDelegate alloc]init];NSApp.delegate=delegate;[NSApp run];
    }catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}
}return 0;}
