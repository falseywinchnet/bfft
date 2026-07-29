#import <Foundation/Foundation.h>
#import <ImageCaptureCore/ImageCaptureCore.h>

static uint16_t read_u16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t read_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static BOOL take_u16(const uint8_t **p, const uint8_t *end, uint16_t *value) {
    if ((size_t)(end - *p) < 2) return NO;
    *value = read_u16(*p);
    *p += 2;
    return YES;
}

static BOOL take_u32(const uint8_t **p, const uint8_t *end, uint32_t *value) {
    if ((size_t)(end - *p) < 4) return NO;
    *value = read_u32(*p);
    *p += 4;
    return YES;
}

static NSString *take_ptp_string(const uint8_t **p, const uint8_t *end) {
    if (*p >= end) return nil;
    uint8_t count = *(*p)++;
    if (count == 0) return @"";
    size_t bytes = (size_t)count * 2;
    if ((size_t)(end - *p) < bytes) return nil;
    size_t contentBytes = count > 0 ? (size_t)(count - 1) * 2 : 0;
    NSString *s = [[NSString alloc] initWithBytes:*p
                                           length:contentBytes
                                         encoding:NSUTF16LittleEndianStringEncoding];
    *p += bytes;
    return s;
}

static NSArray<NSNumber *> *take_u16_array(const uint8_t **p, const uint8_t *end) {
    uint32_t count = 0;
    if (!take_u32(p, end, &count) || count > 65536 ||
        (size_t)(end - *p) < (size_t)count * 2) return nil;
    NSMutableArray<NSNumber *> *values = [NSMutableArray arrayWithCapacity:count];
    for (uint32_t i = 0; i < count; ++i) {
        uint16_t value = 0;
        if (!take_u16(p, end, &value)) return nil;
        [values addObject:@(value)];
    }
    return values;
}

static NSString *hex_list(NSArray<NSNumber *> *values) {
    NSMutableArray<NSString *> *parts = [NSMutableArray arrayWithCapacity:values.count];
    for (NSNumber *n in values) {
        [parts addObject:[NSString stringWithFormat:@"0x%04x", n.unsignedShortValue]];
    }
    return [parts componentsJoinedByString:@" "];
}

static void print_device_info(NSData *data) {
    const uint8_t *p = data.bytes;
    const uint8_t *end = p + data.length;
    uint16_t standardVersion = 0, vendorVersion = 0, functionalMode = 0;
    uint32_t vendorID = 0;
    if (!take_u16(&p, end, &standardVersion) ||
        !take_u32(&p, end, &vendorID) ||
        !take_u16(&p, end, &vendorVersion)) {
        fprintf(stderr, "Truncated DeviceInfo header (%lu bytes)\n",
                (unsigned long)data.length);
        return;
    }
    NSString *vendorDescription = take_ptp_string(&p, end);
    if (!vendorDescription || !take_u16(&p, end, &functionalMode)) {
        fprintf(stderr, "Malformed DeviceInfo vendor section\n");
        return;
    }
    NSArray *operations = take_u16_array(&p, end);
    NSArray *events = take_u16_array(&p, end);
    NSArray *properties = take_u16_array(&p, end);
    NSArray *captureFormats = take_u16_array(&p, end);
    NSArray *imageFormats = take_u16_array(&p, end);
    NSString *manufacturer = take_ptp_string(&p, end);
    NSString *model = take_ptp_string(&p, end);
    NSString *deviceVersion = take_ptp_string(&p, end);
    NSString *serial = take_ptp_string(&p, end);
    if (!operations || !events || !properties || !captureFormats || !imageFormats ||
        !manufacturer || !model || !deviceVersion || !serial) {
        fprintf(stderr, "Malformed DeviceInfo arrays or strings\n");
        return;
    }
    printf("PTP standard: 0x%04x\n", standardVersion);
    printf("Vendor extension: 0x%08x version 0x%04x (%s)\n",
           vendorID, vendorVersion, vendorDescription.UTF8String);
    printf("Functional mode: 0x%04x\n", functionalMode);
    printf("Manufacturer: %s\nModel: %s\nDevice version: %s\nSerial: %s\n",
           manufacturer.UTF8String, model.UTF8String,
           deviceVersion.UTF8String, serial.UTF8String);
    printf("Operations (%lu): %s\n", (unsigned long)operations.count,
           hex_list(operations).UTF8String);
    printf("Events (%lu): %s\n", (unsigned long)events.count,
           hex_list(events).UTF8String);
    printf("Device properties (%lu): %s\n", (unsigned long)properties.count,
           hex_list(properties).UTF8String);
    printf("Capture formats: %s\n", hex_list(captureFormats).UTF8String);
    printf("Image formats: %s\n", hex_list(imageFormats).UTF8String);
    printf("SetUSBMode 0xd15d advertised: %s\n",
           [properties containsObject:@(0xd15d)] ? "yes" : "no");
    printf("USBMode 0xd16e advertised: %s\n",
           [properties containsObject:@(0xd16e)] ? "yes" : "no");
}

@interface XA5Probe : NSObject <ICDeviceBrowserDelegate, ICCameraDeviceDelegate>
@property(nonatomic, strong) ICDeviceBrowser *browser;
@property(nonatomic, strong) ICCameraDevice *camera;
@property(nonatomic) BOOL describeSetUSBMode;
@property(nonatomic) BOOL inspectUSBMode;
@property(nonatomic) BOOL finished;
@end

@implementation XA5Probe

- (void)finish:(int)status {
    if (self.finished) return;
    self.finished = YES;
    [self.browser stop];
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 100 * NSEC_PER_MSEC),
                   dispatch_get_main_queue(), ^{
        exit(status);
    });
}

- (void)closeWithStatus:(int)status {
    [self.camera requestCloseSession];
    [self finish:status];
}

- (void)requestSetUSBModeDescriptor {
    uint8_t packet[16] = {
        16, 0, 0, 0,       // container length
        1, 0,              // command container
        0x14, 0x10,        // GetDevicePropDesc (0x1014)
        2, 0, 0, 0,        // transaction id
        0x5d, 0xd1, 0, 0   // SetUSBMode property (0xd15d)
    };
    NSData *command = [NSData dataWithBytes:packet length:sizeof(packet)];
    [self.camera requestSendPTPCommand:command
                              outData:nil
                           completion:^(NSData *responseData,
                                        NSData *ptpResponseData,
                                        NSError *commandError) {
        if (commandError) {
            fprintf(stderr, "GetDevicePropDesc(0xd15d) transport failed: %s\n",
                    commandError.description.UTF8String);
            [self closeWithStatus:3];
            return;
        }
        if (ptpResponseData.length < 8) {
            fprintf(stderr, "GetDevicePropDesc(0xd15d) returned a short response\n");
            [self closeWithStatus:3];
            return;
        }
        const uint8_t *r = ptpResponseData.bytes;
        printf("GetDevicePropDesc(0xd15d) response: 0x%04x; data bytes: %lu\n",
               read_u16(r + 6), (unsigned long)responseData.length);
        [self closeWithStatus:0];
    }];
}

- (void)requestUSBModeValue {
    uint8_t packet[16] = {
        16, 0, 0, 0,       // container length
        1, 0,              // command container
        0x15, 0x10,        // GetDevicePropValue (0x1015)
        3, 0, 0, 0,        // transaction id
        0x6e, 0xd1, 0, 0   // USBMode property (0xd16e)
    };
    NSData *command = [NSData dataWithBytes:packet length:sizeof(packet)];
    [self.camera requestSendPTPCommand:command
                              outData:nil
                           completion:^(NSData *responseData,
                                        NSData *ptpResponseData,
                                        NSError *commandError) {
        if (commandError) {
            fprintf(stderr, "GetDevicePropValue(0xd16e) transport failed: %s\n",
                    commandError.description.UTF8String);
            [self closeWithStatus:3];
            return;
        }
        if (ptpResponseData.length < 8) {
            fprintf(stderr, "GetDevicePropValue(0xd16e) returned a short response\n");
            [self closeWithStatus:3];
            return;
        }
        const uint8_t *r = ptpResponseData.bytes;
        uint16_t response = read_u16(r + 6);
        printf("GetDevicePropValue(0xd16e) response: 0x%04x; data bytes: %lu",
               response, (unsigned long)responseData.length);
        if (response == 0x2001 && responseData.length == 2) {
            printf("; value: %u", read_u16(responseData.bytes));
        } else if (response == 0x2001 && responseData.length == 4) {
            printf("; value: %u", read_u32(responseData.bytes));
        }
        printf("\n");
        [self closeWithStatus:0];
    }];
}

- (void)requestUSBModeDescriptor {
    uint8_t packet[16] = {
        16, 0, 0, 0,       // container length
        1, 0,              // command container
        0x14, 0x10,        // GetDevicePropDesc (0x1014)
        2, 0, 0, 0,        // transaction id
        0x6e, 0xd1, 0, 0   // USBMode property (0xd16e)
    };
    NSData *command = [NSData dataWithBytes:packet length:sizeof(packet)];
    [self.camera requestSendPTPCommand:command
                              outData:nil
                           completion:^(NSData *responseData,
                                        NSData *ptpResponseData,
                                        NSError *commandError) {
        if (commandError) {
            fprintf(stderr, "GetDevicePropDesc(0xd16e) transport failed: %s\n",
                    commandError.description.UTF8String);
            [self closeWithStatus:3];
            return;
        }
        if (ptpResponseData.length < 8) {
            fprintf(stderr, "GetDevicePropDesc(0xd16e) returned a short response\n");
            [self closeWithStatus:3];
            return;
        }
        const uint8_t *r = ptpResponseData.bytes;
        printf("GetDevicePropDesc(0xd16e) response: 0x%04x; data bytes: %lu\n",
               read_u16(r + 6), (unsigned long)responseData.length);
        [self requestUSBModeValue];
    }];
}

- (void)deviceBrowser:(ICDeviceBrowser *)browser
         didAddDevice:(ICDevice *)device
           moreComing:(BOOL)moreComing {
    if (![device isKindOfClass:[ICCameraDevice class]]) return;
    ICCameraDevice *camera = (ICCameraDevice *)device;
    if (camera.usbVendorID != 0x04cb || camera.usbProductID != 0x02d5) return;
    self.camera = camera;
    camera.delegate = self;
    fprintf(stderr, "Found %s (%04x:%04x); opening read-only PTP session\n",
            camera.name.UTF8String, camera.usbVendorID, camera.usbProductID);
    [camera requestOpenSession];
}

- (void)deviceBrowser:(ICDeviceBrowser *)browser
      didRemoveDevice:(ICDevice *)device
            moreGoing:(BOOL)moreGoing {
}

- (void)device:(ICDevice *)device didOpenSessionWithError:(NSError *)error {
    if (error) {
        fprintf(stderr, "Open session failed: %s\n", error.description.UTF8String);
        [self finish:2];
        return;
    }
    uint8_t packet[12] = {
        12, 0, 0, 0,       // container length
        1, 0,              // command container
        1, 0x10,           // GetDeviceInfo (0x1001)
        1, 0, 0, 0         // transaction id
    };
    NSData *command = [NSData dataWithBytes:packet length:sizeof(packet)];
    [self.camera requestSendPTPCommand:command
                              outData:nil
                           completion:^(NSData *responseData,
                                        NSData *ptpResponseData,
                                        NSError *commandError) {
        if (commandError) {
            fprintf(stderr, "GetDeviceInfo failed: %s\n",
                    commandError.description.UTF8String);
        } else {
            // ImageCaptureCore names the returned data phase responseData.
            print_device_info(responseData);
            if (ptpResponseData.length >= 8) {
                const uint8_t *r = ptpResponseData.bytes;
                printf("PTP response code: 0x%04x\n", read_u16(r + 6));
            }
        }
        if (!commandError && self.inspectUSBMode) {
            [self requestUSBModeDescriptor];
        } else if (!commandError && self.describeSetUSBMode) {
            [self requestSetUSBModeDescriptor];
        } else {
            [self closeWithStatus:commandError ? 3 : 0];
        }
    }];
}

- (void)device:(ICDevice *)device didCloseSessionWithError:(NSError *)error {}
- (void)didRemoveDevice:(ICDevice *)device { [self finish:4]; }
- (void)cameraDevice:(ICCameraDevice *)camera didAddItems:(NSArray *)items {}
- (void)cameraDevice:(ICCameraDevice *)camera didRemoveItems:(NSArray *)items {}
- (void)cameraDevice:(ICCameraDevice *)camera didReceiveThumbnail:(CGImageRef)thumbnail
             forItem:(ICCameraItem *)item error:(NSError *)error {}
- (void)cameraDevice:(ICCameraDevice *)camera didReceiveMetadata:(NSDictionary *)metadata
             forItem:(ICCameraItem *)item error:(NSError *)error {}
- (void)cameraDevice:(ICCameraDevice *)camera didRenameItems:(NSArray *)items {}
- (void)cameraDeviceDidChangeCapability:(ICCameraDevice *)camera {}
- (void)cameraDevice:(ICCameraDevice *)camera didReceivePTPEvent:(NSData *)eventData {}
- (void)deviceDidBecomeReadyWithCompleteContentCatalog:(ICCameraDevice *)device {}
- (void)cameraDeviceDidRemoveAccessRestriction:(ICDevice *)device {}
- (void)cameraDeviceDidEnableAccessRestriction:(ICDevice *)device {}

@end

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        XA5Probe *probe = [XA5Probe new];
        if (argc == 2 &&
            strcmp(argv[1], "--describe-set-usb-mode") == 0) {
            probe.describeSetUSBMode = YES;
        } else if (argc == 2 &&
                   strcmp(argv[1], "--inspect-usb-mode") == 0) {
            probe.inspectUSBMode = YES;
        } else if (argc != 1) {
            fprintf(stderr,
                    "usage: %s [--describe-set-usb-mode|--inspect-usb-mode]\n",
                    argv[0]);
            return 64;
        }
        probe.browser = [ICDeviceBrowser new];
        probe.browser.delegate = probe;
        probe.browser.browsedDeviceTypeMask =
            ICDeviceTypeMaskCamera | ICDeviceLocationTypeMaskLocal;
        [probe.browser start];
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 15 * NSEC_PER_SEC),
                       dispatch_get_main_queue(), ^{
            if (!probe.finished) {
                fprintf(stderr, "Timed out waiting for X-A5\n");
                [probe finish:5];
            }
        });
        [[NSRunLoop mainRunLoop] run];
    }
    return 0;
}
