#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>

#include <signal.h>

static NSMutableDictionary<NSNumber *, NSDictionary *> *seen;

static NSNumber *number_property(io_registry_entry_t service, CFStringRef key) {
    CFTypeRef value = IORegistryEntryCreateCFProperty(
        service, key, kCFAllocatorDefault, 0);
    if (!value) return nil;
    id object = CFBridgingRelease(value);
    return [object isKindOfClass:[NSNumber class]] ? object : nil;
}

static NSString *string_property(io_registry_entry_t service, CFStringRef key) {
    CFTypeRef value = IORegistryEntryCreateCFProperty(
        service, key, kCFAllocatorDefault, 0);
    if (!value) return nil;
    id object = CFBridgingRelease(value);
    return [object isKindOfClass:[NSString class]] ? object : nil;
}

static NSDictionary *identity(io_registry_entry_t service) {
    NSNumber *vendor = number_property(service, CFSTR("idVendor"));
    NSNumber *product = number_property(service, CFSTR("idProduct"));
    if (vendor.unsignedIntValue != 0x04cb) return nil;

    uint64_t registryID = 0;
    IORegistryEntryGetRegistryEntryID(service, &registryID);
    NSString *name = string_property(service, CFSTR("USB Product Name")) ?: @"";
    NSNumber *location = number_property(service, CFSTR("locationID")) ?: @0;
    return @{
        @"registryID": @(registryID),
        @"vendor": vendor ?: @0,
        @"product": product ?: @0,
        @"name": name,
        @"location": location,
    };
}

static void print_event(const char *event, NSDictionary *device) {
    NSTimeInterval wall = NSDate.date.timeIntervalSince1970;
    printf("%.6f %-6s %04x:%04x location=%08x registry=%llu name=\"%s\"\n",
           wall,
           event,
           [device[@"vendor"] unsignedIntValue],
           [device[@"product"] unsignedIntValue],
           [device[@"location"] unsignedIntValue],
           [device[@"registryID"] unsignedLongLongValue],
           [device[@"name"] UTF8String]);
    fflush(stdout);
}

static void appeared(void *refcon, io_iterator_t iterator) {
    (void)refcon;
    io_registry_entry_t service = IO_OBJECT_NULL;
    while ((service = IOIteratorNext(iterator))) {
        @autoreleasepool {
            NSDictionary *device = identity(service);
            if (device) {
                seen[device[@"registryID"]] = device;
                print_event("ADD", device);
            }
        }
        IOObjectRelease(service);
    }
}

static void removed(void *refcon, io_iterator_t iterator) {
    (void)refcon;
    io_registry_entry_t service = IO_OBJECT_NULL;
    while ((service = IOIteratorNext(iterator))) {
        @autoreleasepool {
            uint64_t registryID = 0;
            IORegistryEntryGetRegistryEntryID(service, &registryID);
            NSNumber *key = @(registryID);
            NSDictionary *device = seen[key] ?: identity(service);
            if (device) {
                print_event("REMOVE", device);
                [seen removeObjectForKey:key];
            }
        }
        IOObjectRelease(service);
    }
}

static void stop_trace(int signalNumber) {
    (void)signalNumber;
    CFRunLoopStop(CFRunLoopGetMain());
}

int main(void) {
    @autoreleasepool {
        seen = [NSMutableDictionary dictionary];
        signal(SIGINT, stop_trace);
        signal(SIGTERM, stop_trace);

        IONotificationPortRef port = IONotificationPortCreate(kIOMainPortDefault);
        if (!port) {
            fprintf(stderr, "Unable to create IOKit notification port\n");
            return 2;
        }
        CFRunLoopSourceRef source = IONotificationPortGetRunLoopSource(port);
        CFRunLoopAddSource(CFRunLoopGetMain(), source, kCFRunLoopDefaultMode);

        io_iterator_t addIterator = IO_OBJECT_NULL;
        kern_return_t status = IOServiceAddMatchingNotification(
            port,
            kIOFirstMatchNotification,
            IOServiceMatching("IOUSBHostDevice"),
            appeared,
            NULL,
            &addIterator);
        if (status != KERN_SUCCESS) {
            fprintf(stderr, "Unable to register USB-add notification: 0x%x\n", status);
            return 3;
        }
        appeared(NULL, addIterator);

        io_iterator_t removeIterator = IO_OBJECT_NULL;
        status = IOServiceAddMatchingNotification(
            port,
            kIOTerminatedNotification,
            IOServiceMatching("IOUSBHostDevice"),
            removed,
            NULL,
            &removeIterator);
        if (status != KERN_SUCCESS) {
            fprintf(stderr, "Unable to register USB-remove notification: 0x%x\n", status);
            return 4;
        }
        removed(NULL, removeIterator);

        fprintf(stderr, "Passive Fujifilm USB boot trace armed; press Ctrl-C to stop.\n");
        CFRunLoopRun();

        IOObjectRelease(addIterator);
        IOObjectRelease(removeIterator);
        IONotificationPortDestroy(port);
    }
    return 0;
}
