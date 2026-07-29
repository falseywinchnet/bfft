#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <Foundation/Foundation.h>

static NSString *fourcc(OSType value)
{
	char text[5] = {
		(char)((value >> 24) & 0xff),
		(char)((value >> 16) & 0xff),
		(char)((value >> 8) & 0xff),
		(char)(value & 0xff),
		0,
	};
	for (int index = 0; index < 4; ++index)
		if (text[index] < 32 || text[index] > 126)
			return [NSString stringWithFormat:@"0x%08x", value];
	return [NSString stringWithUTF8String:text];
}

static NSString *dimensions(CMFormatDescriptionRef description)
{
	const CMVideoDimensions size =
		CMVideoFormatDescriptionGetDimensions(description);
	return [NSString stringWithFormat:@"%dx%d", size.width, size.height];
}

static NSString *frame_rates(AVCaptureDeviceFormat *format)
{
	NSMutableArray<NSString *> *values = [NSMutableArray array];
	for (AVFrameRateRange *range in format.videoSupportedFrameRateRanges)
		[values addObject:[NSString
					 stringWithFormat:@"%.3g-%.3g",
							  range.minFrameRate,
							  range.maxFrameRate]];
	return [values componentsJoinedByString:@","];
}

static NSString *exposure_modes(AVCaptureDevice *device)
{
	NSMutableArray<NSString *> *values = [NSMutableArray array];
	if ([device isExposureModeSupported:AVCaptureExposureModeLocked])
		[values addObject:@"locked"];
	if ([device isExposureModeSupported:AVCaptureExposureModeAutoExpose])
		[values addObject:@"auto_once"];
	if ([device
		    isExposureModeSupported:AVCaptureExposureModeContinuousAutoExposure])
		[values addObject:@"continuous_auto"];
	if ([device isExposureModeSupported:AVCaptureExposureModeCustom])
		[values addObject:@"custom_mode_only"];
	return values.count ? [values componentsJoinedByString:@","] : @"none";
}

int main()
{
	@autoreleasepool {
		printf("camera_authorization=%ld\n",
		       (long)[AVCaptureDevice
			       authorizationStatusForMediaType:AVMediaTypeVideo]);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
		NSArray<AVCaptureDevice *> *devices =
			[AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
#pragma clang diagnostic pop

		for (AVCaptureDevice *device in devices) {
			printf("\ndevice=%s\n", device.localizedName.UTF8String);
			printf("  id=%s\n", device.uniqueID.UTF8String);
			printf("  model=%s\n", device.modelID.UTF8String);
			printf("  manufacturer=%s\n",
			       device.manufacturer.UTF8String);
			printf("  transport=%s\n",
			       fourcc((OSType)device.transportType).UTF8String);
			printf("  in_use_elsewhere=%s\n",
			       device.inUseByAnotherApplication ? "true" : "false");
			printf("  exposure_modes=%s\n",
			       exposure_modes(device).UTF8String);
			printf("  exposure_point=%s\n",
			       device.exposurePointOfInterestSupported ? "true"
								       : "false");
			CMFormatDescriptionRef active =
				device.activeFormat.formatDescription;
			printf("  active=%s %s\n",
			       dimensions(active).UTF8String,
			       fourcc(CMFormatDescriptionGetMediaSubType(active))
				       .UTF8String);
			puts("  device_formats:");
			NSUInteger index = 0;
			for (AVCaptureDeviceFormat *format in device.formats) {
				CMFormatDescriptionRef description =
					format.formatDescription;
				printf("    [%lu] %s %s fps=%s\n",
				       (unsigned long)index++,
				       dimensions(description).UTF8String,
				       fourcc(CMFormatDescriptionGetMediaSubType(
						      description))
					       .UTF8String,
				       frame_rates(format).UTF8String);
			}

			AVCaptureSession *session = [[AVCaptureSession alloc] init];
			NSError *error = nil;
			AVCaptureDeviceInput *input =
				[AVCaptureDeviceInput deviceInputWithDevice:device
								     error:&error];
			if (!input || error) {
				printf("  session_error=%s\n",
				       error.localizedDescription.UTF8String);
				continue;
			}
			if (![session canAddInput:input]) {
				puts("  session_input=unsupported");
				continue;
			}
			[session addInput:input];

			AVCaptureVideoDataOutput *video =
				[[AVCaptureVideoDataOutput alloc] init];
			if ([session canAddOutput:video]) {
				[session addOutput:video];
				NSMutableArray<NSString *> *formats =
					[NSMutableArray array];
				for (NSNumber *number in
				     video.availableVideoCVPixelFormatTypes) {
					const OSType value =
						(OSType)number.unsignedIntValue;
					[formats addObject:[NSString
								 stringWithFormat:
									 @"%@(0x%x)",
									 fourcc(value),
									 value]];
				}
				printf("  video_output_pixel_formats=%s\n",
				       [formats componentsJoinedByString:@", "]
					       .UTF8String);
			} else {
				puts("  video_output=unsupported");
			}

			puts("  raw_photo_pixel_formats=unavailable_on_macos");
		}
	}
	return 0;
}
