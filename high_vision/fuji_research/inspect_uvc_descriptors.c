#include <libusb.h>

#include <stdint.h>
#include <stdio.h>

enum {
    MACROSILICON_VENDOR_ID = 0x534d,
    MACROSILICON_PRODUCT_ID = 0x2109,
    USB_CLASS_VIDEO = 0x0e,
    UVC_SC_VIDEOCONTROL = 0x01,
    UVC_CS_INTERFACE = 0x24,
    UVC_VC_EXTENSION_UNIT = 0x06,
};

static void print_hex(const unsigned char *bytes, int length) {
    for (int index = 0; index < length; ++index) {
        printf("%s%02x", index == 0 ? "" : " ", bytes[index]);
    }
    putchar('\n');
}

int main(void) {
    libusb_context *context = NULL;
    libusb_device **devices = NULL;
    int result = libusb_init(&context);
    if (result != LIBUSB_SUCCESS) {
        fprintf(stderr, "libusb_init: %s\n", libusb_error_name(result));
        return 1;
    }

    ssize_t count = libusb_get_device_list(context, &devices);
    if (count < 0) {
        fprintf(stderr, "libusb_get_device_list: %s\n",
                libusb_error_name((int)count));
        libusb_exit(context);
        return 1;
    }

    int found = 0;
    for (ssize_t device_index = 0; device_index < count; ++device_index) {
        libusb_device *device = devices[device_index];
        struct libusb_device_descriptor device_descriptor;
        if (libusb_get_device_descriptor(device, &device_descriptor) !=
            LIBUSB_SUCCESS) {
            continue;
        }
        if (device_descriptor.idVendor != MACROSILICON_VENDOR_ID ||
            device_descriptor.idProduct != MACROSILICON_PRODUCT_ID) {
            continue;
        }
        found = 1;
        printf("MacroSilicon %04x:%04x, bus %u address %u\n",
               device_descriptor.idVendor, device_descriptor.idProduct,
               libusb_get_bus_number(device), libusb_get_device_address(device));

        struct libusb_config_descriptor *configuration = NULL;
        result = libusb_get_active_config_descriptor(device, &configuration);
        if (result != LIBUSB_SUCCESS) {
            fprintf(stderr, "active configuration: %s\n",
                    libusb_error_name(result));
            continue;
        }

        int extension_units = 0;
        for (uint8_t interface_index = 0;
             interface_index < configuration->bNumInterfaces;
             ++interface_index) {
            const struct libusb_interface *interface =
                &configuration->interface[interface_index];
            for (int alternate_index = 0;
                 alternate_index < interface->num_altsetting;
                 ++alternate_index) {
                const struct libusb_interface_descriptor *alternate =
                    &interface->altsetting[alternate_index];
                if (alternate->bInterfaceClass != USB_CLASS_VIDEO ||
                    alternate->bInterfaceSubClass != UVC_SC_VIDEOCONTROL) {
                    continue;
                }
                printf("VideoControl interface %u alt %u, %d extra bytes\n",
                       alternate->bInterfaceNumber, alternate->bAlternateSetting,
                       alternate->extra_length);

                int offset = 0;
                while (offset + 3 <= alternate->extra_length) {
                    const unsigned char *descriptor = alternate->extra + offset;
                    int length = descriptor[0];
                    if (length < 3 || offset + length > alternate->extra_length) {
                        fprintf(stderr,
                                "malformed class descriptor at offset %d\n",
                                offset);
                        break;
                    }
                    printf("  subtype 0x%02x length %d: ", descriptor[2],
                           length);
                    print_hex(descriptor, length);
                    if (descriptor[1] == UVC_CS_INTERFACE &&
                        descriptor[2] == UVC_VC_EXTENSION_UNIT) {
                        ++extension_units;
                        if (length >= 21) {
                            printf("    extension unit id=%u guid=", descriptor[3]);
                            print_hex(descriptor + 4, 16);
                        }
                    }
                    offset += length;
                }
            }
        }
        printf("Vendor extension units: %d\n", extension_units);
        libusb_free_config_descriptor(configuration);
    }

    libusb_free_device_list(devices, 1);
    libusb_exit(context);
    if (!found) {
        fprintf(stderr, "MacroSilicon 534d:2109 was not found\n");
        return 2;
    }
    return 0;
}
