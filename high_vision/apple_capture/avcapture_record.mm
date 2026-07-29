#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

@interface HVFrameWriter
    : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate> {
  FILE *_video;
  FILE *_timestamps;
  int _width;
  int _height;
  int _targetFrames;
  int _frames;
  dispatch_semaphore_t _done;
}
- (instancetype)initWithVideo:(FILE *)video
                   timestamps:(FILE *)timestamps
                        width:(int)width
                       height:(int)height
                 targetFrames:(int)targetFrames
                         done:(dispatch_semaphore_t)done;
- (int)frames;
@end

@implementation HVFrameWriter
- (instancetype)initWithVideo:(FILE *)video
                   timestamps:(FILE *)timestamps
                        width:(int)width
                       height:(int)height
                 targetFrames:(int)targetFrames
                         done:(dispatch_semaphore_t)done {
  self = [super init];
  if (self) {
    _video = video;
    _timestamps = timestamps;
    _width = width;
    _height = height;
    _targetFrames = targetFrames;
    _frames = 0;
    _done = done;
  }
  return self;
}

- (int)frames {
  return _frames;
}

- (void)captureOutput:(AVCaptureOutput *)output
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection *)connection {
  (void)output;
  (void)connection;
  @autoreleasepool {
    if (_frames >= _targetFrames) {
      return;
    }
    CVPixelBufferRef pixelBuffer =
        CMSampleBufferGetImageBuffer(sampleBuffer);
    if (!pixelBuffer ||
        CVPixelBufferGetPixelFormatType(pixelBuffer) !=
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ||
        CVPixelBufferGetPlaneCount(pixelBuffer) != 2 ||
        CVPixelBufferGetWidth(pixelBuffer) != (size_t)_width ||
        CVPixelBufferGetHeight(pixelBuffer) != (size_t)_height) {
      std::fprintf(stderr, "unexpected pixel buffer format or dimensions\n");
      dispatch_semaphore_signal(_done);
      return;
    }

    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    const auto *y = static_cast<const unsigned char *>(
        CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0));
    const auto *uv = static_cast<const unsigned char *>(
        CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1));
    const size_t yStride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);
    const size_t uvStride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1);
    for (int row = 0; row < _height; ++row) {
      std::fwrite(y + row * yStride, 1, (size_t)_width, _video);
    }
    for (int row = 0; row < _height / 2; ++row) {
      std::fwrite(uv + row * uvStride, 1, (size_t)_width, _video);
    }
    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

    const double pts =
        CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer));
    const double duration =
        CMTimeGetSeconds(CMSampleBufferGetDuration(sampleBuffer));
    std::fprintf(_timestamps, "%d,%.9f,%.9f\n", _frames, pts, duration);
    ++_frames;
    if (_frames == _targetFrames) {
      std::fflush(_video);
      std::fflush(_timestamps);
      dispatch_semaphore_signal(_done);
    }
  }
}
@end

static void usage(const char *program) {
  std::fprintf(
      stderr,
      "usage: %s OUTPUT_PREFIX WIDTH HEIGHT FPS SECONDS [DEVICE_MATCH]\n",
      program);
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc < 6 || argc > 7) {
      usage(argv[0]);
      return 2;
    }
    NSString *prefix = [NSString stringWithUTF8String:argv[1]];
    const int width = std::atoi(argv[2]);
    const int height = std::atoi(argv[3]);
    const double fps = std::atof(argv[4]);
    const double seconds = std::atof(argv[5]);
    NSString *deviceMatch =
        argc == 7 ? [NSString stringWithUTF8String:argv[6]]
                  : @"USB3.0 HD VIDEO";
    if (width <= 0 || height <= 0 || fps <= 0.0 || seconds <= 0.0) {
      usage(argv[0]);
      return 2;
    }

    AVCaptureDevice *device = nil;
    AVCaptureDeviceDiscoverySession *discovery =
        [AVCaptureDeviceDiscoverySession
            discoverySessionWithDeviceTypes:@[
              AVCaptureDeviceTypeExternal
            ]
                                  mediaType:AVMediaTypeVideo
                                   position:AVCaptureDevicePositionUnspecified];
    for (AVCaptureDevice *candidate in discovery.devices) {
      if ([candidate.localizedName
              rangeOfString:deviceMatch
                    options:NSCaseInsensitiveSearch].location != NSNotFound) {
        device = candidate;
        break;
      }
    }
    if (!device) {
      std::fprintf(stderr, "capture device not found: %s\n", argv[6]);
      return 3;
    }

    AVCaptureDeviceFormat *selected = nil;
    AVFrameRateRange *selectedRange = nil;
    for (AVCaptureDeviceFormat *format in device.formats) {
      CMVideoDimensions dimensions =
          CMVideoFormatDescriptionGetDimensions(format.formatDescription);
      FourCharCode subtype =
          CMFormatDescriptionGetMediaSubType(format.formatDescription);
      if (dimensions.width != width || dimensions.height != height ||
          subtype != kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange) {
        continue;
      }
      for (AVFrameRateRange *range in format.videoSupportedFrameRateRanges) {
        if (fps >= range.minFrameRate - 0.01 &&
            fps <= range.maxFrameRate + 0.01) {
          selected = format;
          selectedRange = range;
          break;
        }
      }
      if (selected) {
        break;
      }
    }
    if (!selected) {
      std::fprintf(stderr, "no native %dx%d 420v format at %.6f fps\n",
                   width, height, fps);
      return 4;
    }

    NSError *error = nil;
    AVCaptureSession *session = [[AVCaptureSession alloc] init];
    if (width == 1920 && height == 1080) {
      session.sessionPreset = AVCaptureSessionPreset1920x1080;
    } else if (width == 1280 && height == 720) {
      session.sessionPreset = AVCaptureSessionPreset1280x720;
    } else {
      session.sessionPreset = AVCaptureSessionPresetHigh;
    }
    AVCaptureDeviceInput *input =
        [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
    if (!input || ![session canAddInput:input]) {
      std::fprintf(stderr, "cannot create capture input: %s\n",
                   error.localizedDescription.UTF8String);
      return 6;
    }
    [session addInput:input];

    AVCaptureVideoDataOutput *output =
        [[AVCaptureVideoDataOutput alloc] init];
    output.videoSettings = @{
      (id)kCVPixelBufferPixelFormatTypeKey :
          @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
    };
    output.alwaysDiscardsLateVideoFrames = NO;
    if (![session canAddOutput:output]) {
      std::fprintf(stderr, "cannot add video data output\n");
      return 7;
    }
    [session addOutput:output];

    // Session preset negotiation can reset the device format on macOS. Apply
    // the driver's exact advertised CMTime values only after the graph exists.
    if (![device lockForConfiguration:&error]) {
      std::fprintf(stderr, "cannot lock device: %s\n",
                   error.localizedDescription.UTF8String);
      return 5;
    }
    device.activeFormat = selected;
    device.activeVideoMinFrameDuration = selectedRange.minFrameDuration;
    device.activeVideoMaxFrameDuration = selectedRange.maxFrameDuration;
    [device unlockForConfiguration];

    NSString *videoPath = [prefix stringByAppendingString:@".nv12"];
    NSString *timestampPath = [prefix stringByAppendingString:@".csv"];
    FILE *video = std::fopen(videoPath.fileSystemRepresentation, "wb");
    FILE *timestamps =
        std::fopen(timestampPath.fileSystemRepresentation, "w");
    if (!video || !timestamps) {
      std::fprintf(stderr, "cannot open output files at %s\n", argv[1]);
      if (video) std::fclose(video);
      if (timestamps) std::fclose(timestamps);
      return 8;
    }
    std::fprintf(timestamps, "frame,pts_seconds,duration_seconds\n");

    const int targetFrames = (int)std::ceil(fps * seconds);
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    dispatch_queue_t queue =
        dispatch_queue_create("high_vision.capture", DISPATCH_QUEUE_SERIAL);
    HVFrameWriter *writer =
        [[HVFrameWriter alloc] initWithVideo:video
                                  timestamps:timestamps
                                       width:width
                                      height:height
                                targetFrames:targetFrames
                                        done:done];
    [output setSampleBufferDelegate:writer queue:queue];
    [session startRunning];
    const int64_t timeoutNanoseconds =
        (int64_t)((seconds + 10.0) * (double)NSEC_PER_SEC);
    long waitResult = dispatch_semaphore_wait(
        done, dispatch_time(DISPATCH_TIME_NOW, timeoutNanoseconds));
    [session stopRunning];
    [output setSampleBufferDelegate:nil queue:nil];
    dispatch_sync(queue, ^{});
    std::fclose(video);
    std::fclose(timestamps);

    std::printf(
        "device=%s format=%dx%d 420v requested_fps=%.6f frames=%d "
        "video=%s timestamps=%s\n",
        device.localizedName.UTF8String, width, height, fps, writer.frames,
        videoPath.fileSystemRepresentation,
        timestampPath.fileSystemRepresentation);
    return waitResult == 0 && writer.frames == targetFrames ? 0 : 9;
  }
}
