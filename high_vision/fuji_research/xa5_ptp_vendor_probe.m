#import <Foundation/Foundation.h>
#import <ImageCaptureCore/ImageCaptureCore.h>

static uint16_t read_u16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t read_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void append_u16(NSMutableData *data, uint16_t value) {
    uint8_t bytes[2] = {
        (uint8_t)(value & 0xff),
        (uint8_t)((value >> 8) & 0xff),
    };
    [data appendBytes:bytes length:sizeof(bytes)];
}

static void append_u32(NSMutableData *data, uint32_t value) {
    uint8_t bytes[4] = {
        (uint8_t)(value & 0xff),
        (uint8_t)((value >> 8) & 0xff),
        (uint8_t)((value >> 16) & 0xff),
        (uint8_t)((value >> 24) & 0xff),
    };
    [data appendBytes:bytes length:sizeof(bytes)];
}

static NSData *parse_hex_data(NSString *text) {
    NSMutableString *clean = [NSMutableString string];
    NSCharacterSet *hex = [NSCharacterSet characterSetWithCharactersInString:
                           @"0123456789abcdefABCDEF"];
    for (NSUInteger i = 0; i < text.length; ++i) {
        unichar c = [text characterAtIndex:i];
        if ([hex characterIsMember:c]) {
            [clean appendFormat:@"%C", c];
        }
    }
    if (clean.length % 2 != 0) return nil;
    NSMutableData *result = [NSMutableData dataWithCapacity:clean.length / 2];
    for (NSUInteger i = 0; i < clean.length; i += 2) {
        NSString *pair = [clean substringWithRange:NSMakeRange(i, 2)];
        unsigned int byte = 0;
        if (![[NSScanner scannerWithString:pair] scanHexInt:&byte]) return nil;
        uint8_t value = (uint8_t)byte;
        [result appendBytes:&value length:1];
    }
    return result;
}

static void print_hex(NSData *data) {
    const uint8_t *bytes = data.bytes;
    for (NSUInteger offset = 0; offset < data.length; offset += 16) {
        printf("%04lx:", (unsigned long)offset);
        NSUInteger count = MIN((NSUInteger)16, data.length - offset);
        for (NSUInteger i = 0; i < count; ++i) {
            printf(" %02x", bytes[offset + i]);
        }
        printf("\n");
    }
}

@interface XA5VendorProbe : NSObject <ICDeviceBrowserDelegate, ICCameraDeviceDelegate>
@property(nonatomic, strong) ICDeviceBrowser *browser;
@property(nonatomic, strong) ICCameraDevice *camera;
@property(nonatomic) uint16_t operation;
@property(nonatomic, strong) NSArray<NSNumber *> *parameters;
@property(nonatomic, strong) NSData *outData;
@property(nonatomic, strong) NSArray<NSDictionary *> *sequence;
@property(nonatomic, strong) NSMutableArray<NSNumber *> *responseTransactions;
@property(nonatomic) NSUInteger sequenceIndex;
@property(nonatomic) BOOL holding;
@property(nonatomic) BOOL closing;
@property(nonatomic) int pendingExitStatus;
@property(nonatomic) BOOL finished;
- (void)closeWithStatus:(int)status;
@end

@implementation XA5VendorProbe

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
    if (self.closing) return;
    self.closing = YES;
    self.pendingExitStatus = status;
    [self.camera requestCloseSession];
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC),
                   dispatch_get_main_queue(), ^{
        if (!self.finished) [self finish:self.pendingExitStatus];
    });
}

- (void)sendOperation {
    NSDictionary *step = self.sequence[self.sequenceIndex];
    self.operation = [step[@"operation"] unsignedShortValue];
    NSMutableArray<NSNumber *> *resolvedParameters = [NSMutableArray array];
    for (id parameter in step[@"parameters"]) {
        uint32_t value = 0;
        if ([parameter isKindOfClass:[NSString class]]) {
            NSUInteger responseIndex =
                [(NSString *)parameter substringFromIndex:1].integerValue;
            value = self.responseTransactions[responseIndex].unsignedIntValue;
        } else {
            value = [parameter unsignedIntValue];
        }
        [resolvedParameters addObject:@(value)];
    }
    self.parameters = resolvedParameters;
    self.outData = step[@"outData"];
    NSMutableData *command = [NSMutableData data];
    append_u32(command, (uint32_t)(12 + 4 * self.parameters.count));
    append_u16(command, 1);  // PTP command container.
    append_u16(command, self.operation);
    append_u32(command, 1);  // Transaction id.
    for (NSNumber *parameter in self.parameters) {
        append_u32(command, parameter.unsignedIntValue);
    }

    printf("Sending operation 0x%04x with %lu parameter(s)",
           self.operation, (unsigned long)self.parameters.count);
    for (NSNumber *parameter in self.parameters) {
        printf(" 0x%08x", parameter.unsignedIntValue);
    }
    printf("; out-data bytes: %lu\n", (unsigned long)self.outData.length);

    [self.camera requestSendPTPCommand:command
                              outData:self.outData
                           completion:^(NSData *responseData,
                                        NSData *ptpResponseData,
                                        NSError *commandError) {
        if (commandError) {
            fprintf(stderr, "Transport error: %s\n",
                    commandError.description.UTF8String);
            [self closeWithStatus:3];
            return;
        }
        if (ptpResponseData.length < 12) {
            fprintf(stderr, "Short PTP response: %lu bytes\n",
                    (unsigned long)ptpResponseData.length);
            if (ptpResponseData.length) print_hex(ptpResponseData);
            [self closeWithStatus:3];
            return;
        }
        const uint8_t *response = ptpResponseData.bytes;
        uint16_t code = read_u16(response + 6);
        uint32_t transaction = read_u32(response + 8);
        printf("Response: 0x%04x; transaction: %u; data bytes: %lu\n",
               code, transaction, (unsigned long)responseData.length);
        [self.responseTransactions addObject:@(transaction)];
        if (ptpResponseData.length > 12) {
            printf("Response parameters/container:\n");
            print_hex(ptpResponseData);
        }
        if (responseData.length) {
            printf("Data phase:\n");
            print_hex(responseData);
        }
        if (code != 0x2001 &&
            ![step[@"continueOnError"] boolValue]) {
            [self closeWithStatus:0];
            return;
        }
        self.sequenceIndex += 1;
        if (self.sequenceIndex < self.sequence.count) {
            NSUInteger delay =
                [step[@"delayMilliseconds"] unsignedIntegerValue];
            dispatch_after(
                dispatch_time(DISPATCH_TIME_NOW,
                              (int64_t)delay * NSEC_PER_MSEC),
                dispatch_get_main_queue(), ^{
                    [self sendOperation];
                });
        } else if ([step[@"holdOpen"] boolValue]) {
            self.holding = YES;
            printf("Holding the PTP session open; press Control-C to close it.\n");
        } else {
            [self closeWithStatus:0];
        }
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
    fprintf(stderr, "Found %s (%04x:%04x); opening PTP session\n",
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
    [self sendOperation];
}

- (void)device:(ICDevice *)device didCloseSessionWithError:(NSError *)error {
    if (error) {
        fprintf(stderr, "Close session failed: %s\n",
                error.description.UTF8String);
    }
    [self finish:self.pendingExitStatus];
}

- (void)didRemoveDevice:(ICDevice *)device {
    [self finish:4];
}

- (void)deviceDidBecomeReady:(ICDevice *)device {
}

- (void)device:(ICDevice *)device
 didReceiveStatusInformation:(NSDictionary<ICDeviceStatus, id> *)status {
}

- (void)device:(ICDevice *)device
didEncounterError:(NSError *)error {
    fprintf(stderr, "Device error: %s\n", error.description.UTF8String);
}

- (void)cameraDevice:(ICCameraDevice *)camera
       didAddItems:(NSArray<ICCameraItem *> *)items {
}

- (void)cameraDevice:(ICCameraDevice *)camera
    didRemoveItems:(NSArray<ICCameraItem *> *)items {
}

- (void)cameraDevice:(ICCameraDevice *)camera
didRenameItems:(NSArray<ICCameraItem *> *)items {
}

- (void)cameraDeviceDidChangeCapability:(ICCameraDevice *)camera {
}

- (void)cameraDevice:(ICCameraDevice *)camera
  didReceivePTPEvent:(NSData *)eventData {
    printf("PTP event (%lu bytes):\n", (unsigned long)eventData.length);
    print_hex(eventData);
}

- (void)deviceDidBecomeReadyWithCompleteContentCatalog:(ICCameraDevice *)device {
}

- (void)cameraDeviceDidRemoveAccessRestriction:(ICDevice *)device {
}

- (void)cameraDeviceDidEnableAccessRestriction:(ICDevice *)device {
}

- (void)cameraDevice:(ICCameraDevice *)camera
 didReceiveThumbnail:(CGImageRef)thumbnail
             forItem:(ICCameraItem *)item
               error:(NSError *)error {
}

- (void)cameraDevice:(ICCameraDevice *)camera
didReceiveMetadata:(NSDictionary *)metadata
             forItem:(ICCameraItem *)item
               error:(NSError *)error {
}

- (void)cameraDevice:(ICCameraDevice *)camera
didCompleteDeleteFilesWithError:(NSError *)error {
}

- (void)cameraDevice:(ICCameraDevice *)camera
 didCompleteDownloadWithError:(NSError *)error
             options:(NSDictionary<NSString *, id> *)options
      contextInfo:(void *)contextInfo {
}

@end

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s OPERATION [PARAM ...] [--out-hex HEX]"
            " [--delay-ms MILLISECONDS]"
            " [--continue-on-error]"
            " [--hold-open]"
            " [--next OPERATION [PARAM ...] ...]\\n"
            "  integers accept decimal or 0x-prefixed notation\\n"
            "  @N uses the response transaction id from sequence step N\\n",
            program);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            usage(argv[0]);
            return 64;
        }
        char *end = NULL;
        NSMutableArray<NSDictionary *> *sequence = [NSMutableArray array];
        NSMutableArray<id> *parameters = [NSMutableArray array];
        NSData *outData = nil;
        NSUInteger delayMilliseconds = 0;
        BOOL continueOnError = NO;
        BOOL holdOpen = NO;
        unsigned long operation = 0;
        BOOL haveOperation = NO;
        for (int i = 1; i <= argc; ++i) {
            if (i == argc || strcmp(argv[i], "--next") == 0) {
                if (!haveOperation) {
                    usage(argv[0]);
                    return 64;
                }
                [sequence addObject:@{
                    @"operation": @((uint16_t)operation),
                    @"parameters": [parameters copy],
                    @"outData": outData ?: [NSData data],
                    @"delayMilliseconds": @(delayMilliseconds),
                    @"continueOnError": @(continueOnError),
                    @"holdOpen": @(holdOpen),
                }];
                if (i == argc) break;
                haveOperation = NO;
                parameters = [NSMutableArray array];
                outData = nil;
                delayMilliseconds = 0;
                continueOnError = NO;
                holdOpen = NO;
                continue;
            }
            if (!haveOperation) {
                operation = strtoul(argv[i], &end, 0);
                if (!end || *end || operation > UINT16_MAX) {
                    usage(argv[0]);
                    return 64;
                }
                haveOperation = YES;
                continue;
            }
            if (strcmp(argv[i], "--out-hex") == 0) {
                if (++i >= argc || outData) {
                    usage(argv[0]);
                    return 64;
                }
                outData = parse_hex_data([NSString stringWithUTF8String:argv[i]]);
                if (!outData) {
                    fprintf(stderr, "Invalid hex data\n");
                    return 64;
                }
                continue;
            }
            if (strcmp(argv[i], "--delay-ms") == 0) {
                if (++i >= argc || delayMilliseconds != 0) {
                    usage(argv[0]);
                    return 64;
                }
                unsigned long value = strtoul(argv[i], &end, 0);
                if (!end || *end || value > 10000) {
                    usage(argv[0]);
                    return 64;
                }
                delayMilliseconds = (NSUInteger)value;
                continue;
            }
            if (strcmp(argv[i], "--continue-on-error") == 0) {
                continueOnError = YES;
                continue;
            }
            if (strcmp(argv[i], "--hold-open") == 0) {
                holdOpen = YES;
                continue;
            }
            if (parameters.count >= 5) {
                usage(argv[0]);
                return 64;
            }
            if (argv[i][0] == '@') {
                unsigned long stepIndex = strtoul(argv[i] + 1, &end, 0);
                if (!end || *end || stepIndex >= sequence.count) {
                    usage(argv[0]);
                    return 64;
                }
                [parameters addObject:
                    [NSString stringWithUTF8String:argv[i]]];
            } else {
                unsigned long value = strtoul(argv[i], &end, 0);
                if (!end || *end || value > UINT32_MAX) {
                    usage(argv[0]);
                    return 64;
                }
                [parameters addObject:@((uint32_t)value)];
            }
        }

        XA5VendorProbe *probe = [XA5VendorProbe new];
        probe.sequence = sequence;
        probe.responseTransactions = [NSMutableArray array];
        probe.browser = [ICDeviceBrowser new];
        probe.browser.delegate = probe;
        probe.browser.browsedDeviceTypeMask = ICDeviceTypeMaskCamera |
                                              ICDeviceLocationTypeMaskLocal;
        [probe.browser start];

        signal(SIGINT, SIG_IGN);
        signal(SIGTERM, SIG_IGN);
        dispatch_source_t interruptSource =
            dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, SIGINT, 0,
                                   dispatch_get_main_queue());
        dispatch_source_t terminateSource =
            dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, SIGTERM, 0,
                                   dispatch_get_main_queue());
        dispatch_source_set_event_handler(interruptSource, ^{
            if (probe.camera) {
                [probe closeWithStatus:130];
            } else {
                [probe finish:130];
            }
        });
        dispatch_source_set_event_handler(terminateSource, ^{
            if (probe.camera) {
                [probe closeWithStatus:143];
            } else {
                [probe finish:143];
            }
        });
        dispatch_resume(interruptSource);
        dispatch_resume(terminateSource);

        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC),
                       dispatch_get_main_queue(), ^{
            if (!probe.finished && !probe.holding) {
                fprintf(stderr, "Timed out waiting for X-A5 or response\n");
                [probe finish:4];
            }
        });
        [[NSRunLoop mainRunLoop] run];
    }
    return 0;
}
