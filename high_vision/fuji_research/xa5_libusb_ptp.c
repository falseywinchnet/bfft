#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <libusb-1.0/libusb.h>

#define XA5_VENDOR_ID 0x04cb
#define XA5_PRODUCT_ID 0x02d5

#define PTP_CONTAINER_COMMAND 1
#define PTP_CONTAINER_DATA 2
#define PTP_CONTAINER_RESPONSE 3

#define PTP_OC_OPEN_SESSION 0x1002
#define PTP_OC_CLOSE_SESSION 0x1003
#define PTP_OC_GET_OBJECT_HANDLES 0x1007
#define PTP_OC_GET_OBJECT_INFO 0x1008
#define PTP_OC_GET_OBJECT 0x1009
#define PTP_OC_SEND_OBJECT_INFO 0x100c
#define PTP_OC_SEND_OBJECT 0x100d
#define PTP_OC_GET_DEVICE_PROP_DESC 0x1014
#define PTP_OC_GET_DEVICE_PROP_VALUE 0x1015
#define PTP_OC_SET_DEVICE_PROP_VALUE 0x1016
#define PTP_OC_TERMINATE_OPEN_CAPTURE 0x1018
#define PTP_OC_INITIATE_OPEN_CAPTURE 0x101c
#define PTP_OC_INITIATE_MOVIE_CAPTURE 0x9020
#define PTP_OC_TERMINATE_MOVIE_CAPTURE 0x9021
#define PTP_OC_GET_CAPTURE_PREVIEW 0x9022
#define PTP_OC_CANCEL_INITIATE_CAPTURE 0x9030
#define PTP_RC_OK 0x2001
#define PTP_RC_SESSION_ALREADY_OPEN 0x201e
#define PTP_USB_REQUEST_DEVICE_RESET 0x66
#define PTP_OFC_SCRIPT 0x3002
#define PTP_DPC_FUJI_SET_USB_MODE 0xd15d
#define PTP_DPC_FUJI_VIDEO_OUT_ON_OFF 0xd168
#define PTP_DPC_FUJI_FORCE_MODE 0xd230
#define XA5_INTERNAL_STORAGE_ID 0x00010001

typedef struct {
    uint8_t *bytes;
    uint32_t length;
    uint16_t type;
    uint16_t code;
    uint32_t transaction;
} ptp_container;

static volatile sig_atomic_t interrupted;
static uint8_t ptp_interrupt_in;

static uint16_t get_u16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t get_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void put_u16(uint8_t *p, uint16_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
}

static void put_u32(uint8_t *p, uint32_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
    p[2] = (uint8_t)(value >> 16);
    p[3] = (uint8_t)(value >> 24);
}

static size_t ptp_scalar_size(uint16_t data_type) {
    switch (data_type) {
        case 0x0001:
        case 0x0002:
            return 1;
        case 0x0003:
        case 0x0004:
            return 2;
        case 0x0005:
        case 0x0006:
            return 4;
        case 0x0007:
        case 0x0008:
            return 8;
        default:
            return 0;
    }
}

static void put_scalar(uint8_t *p, size_t size, uint64_t value) {
    for (size_t i = 0; i < size; ++i) {
        p[i] = (uint8_t)(value >> (8 * i));
    }
}

static uint64_t get_scalar(const uint8_t *p, size_t size) {
    uint64_t value = 0;
    for (size_t i = 0; i < size; ++i) {
        value |= (uint64_t)p[i] << (8 * i);
    }
    return value;
}

static void print_hex_bytes(const char *label,
                            const uint8_t *bytes,
                            size_t length) {
    printf("%s (%zu byte(s)):", label, length);
    for (size_t i = 0; i < length; ++i) {
        printf(" %02x", bytes[i]);
    }
    printf("\n");
}

static void on_signal(int signal_number) {
    (void)signal_number;
    interrupted = 1;
}

static const char *transfer_name(uint8_t attributes) {
    switch (attributes & LIBUSB_TRANSFER_TYPE_MASK) {
        case LIBUSB_TRANSFER_TYPE_CONTROL: return "control";
        case LIBUSB_TRANSFER_TYPE_ISOCHRONOUS: return "isochronous";
        case LIBUSB_TRANSFER_TYPE_BULK: return "bulk";
        case LIBUSB_TRANSFER_TYPE_INTERRUPT: return "interrupt";
        default: return "unknown";
    }
}

static int find_ptp_interface(const struct libusb_config_descriptor *config,
                              int *interface_number,
                              uint8_t *bulk_in,
                              uint8_t *bulk_out,
                              uint8_t *interrupt_in) {
    *interface_number = -1;
    *bulk_in = 0;
    *bulk_out = 0;
    *interrupt_in = 0;

    printf("Configuration %u: %u interface(s)\n",
           config->bConfigurationValue, config->bNumInterfaces);
    for (uint8_t i = 0; i < config->bNumInterfaces; ++i) {
        const struct libusb_interface *interface = &config->interface[i];
        for (int a = 0; a < interface->num_altsetting; ++a) {
            const struct libusb_interface_descriptor *alt =
                &interface->altsetting[a];
            printf("  interface %u alt %u: class %02x/%02x/%02x, "
                   "%u endpoint(s)\n",
                   alt->bInterfaceNumber, alt->bAlternateSetting,
                   alt->bInterfaceClass, alt->bInterfaceSubClass,
                   alt->bInterfaceProtocol, alt->bNumEndpoints);
            for (uint8_t e = 0; e < alt->bNumEndpoints; ++e) {
                const struct libusb_endpoint_descriptor *endpoint =
                    &alt->endpoint[e];
                printf("    endpoint 0x%02x: %-11s max-packet %u\n",
                       endpoint->bEndpointAddress,
                       transfer_name(endpoint->bmAttributes),
                       endpoint->wMaxPacketSize);
            }

            if (alt->bInterfaceClass != LIBUSB_CLASS_IMAGE ||
                alt->bInterfaceSubClass != 1 ||
                alt->bInterfaceProtocol != 1) {
                continue;
            }
            *interface_number = alt->bInterfaceNumber;
            for (uint8_t e = 0; e < alt->bNumEndpoints; ++e) {
                const struct libusb_endpoint_descriptor *endpoint =
                    &alt->endpoint[e];
                uint8_t transfer =
                    endpoint->bmAttributes & LIBUSB_TRANSFER_TYPE_MASK;
                uint8_t direction =
                    endpoint->bEndpointAddress & LIBUSB_ENDPOINT_DIR_MASK;
                if (transfer == LIBUSB_TRANSFER_TYPE_BULK &&
                    direction == LIBUSB_ENDPOINT_IN) {
                    *bulk_in = endpoint->bEndpointAddress;
                } else if (transfer == LIBUSB_TRANSFER_TYPE_BULK &&
                           direction == LIBUSB_ENDPOINT_OUT) {
                    *bulk_out = endpoint->bEndpointAddress;
                } else if (transfer == LIBUSB_TRANSFER_TYPE_INTERRUPT &&
                           direction == LIBUSB_ENDPOINT_IN) {
                    *interrupt_in = endpoint->bEndpointAddress;
                }
            }
        }
    }

    return *interface_number >= 0 && *bulk_in != 0 && *bulk_out != 0;
}

static int send_command(libusb_device_handle *handle,
                        uint8_t bulk_out,
                        uint16_t operation,
                        uint32_t transaction,
                        const uint32_t *parameters,
                        size_t parameter_count) {
    if (parameter_count > 5) return LIBUSB_ERROR_INVALID_PARAM;
    uint8_t command[12 + 5 * 4] = {0};
    uint32_t length = (uint32_t)(12 + parameter_count * 4);
    put_u32(command, length);
    put_u16(command + 4, PTP_CONTAINER_COMMAND);
    put_u16(command + 6, operation);
    put_u32(command + 8, transaction);
    for (size_t i = 0; i < parameter_count; ++i) {
        put_u32(command + 12 + i * 4, parameters[i]);
    }

    int transferred = 0;
    int result = libusb_bulk_transfer(handle, bulk_out, command, (int)length,
                                      &transferred, 3000);
    if (result != 0) return result;
    if (transferred != (int)length) return LIBUSB_ERROR_IO;
    printf("sent operation 0x%04x, transaction %" PRIu32 "\n",
           operation, transaction);
    return 0;
}

static int send_data(libusb_device_handle *handle,
                     uint8_t bulk_out,
                     uint16_t operation,
                     uint32_t transaction,
                     const uint8_t *payload,
                     size_t payload_length) {
    if (payload_length > UINT32_MAX - 12U) return LIBUSB_ERROR_INVALID_PARAM;
    uint32_t length = (uint32_t)payload_length + 12U;
    uint8_t *container = malloc(length);
    if (!container) return LIBUSB_ERROR_NO_MEM;
    put_u32(container, length);
    put_u16(container + 4, PTP_CONTAINER_DATA);
    put_u16(container + 6, operation);
    put_u32(container + 8, transaction);
    if (payload_length != 0) {
        memcpy(container + 12, payload, payload_length);
    }

    int transferred = 0;
    int result = libusb_bulk_transfer(handle, bulk_out, container, (int)length,
                                      &transferred, 3000);
    free(container);
    if (result != 0) return result;
    if (transferred != (int)length) return LIBUSB_ERROR_IO;
    printf("sent %" PRIu32 "-byte data container for operation 0x%04x\n",
           length, operation);
    return 0;
}

static void free_container(ptp_container *container) {
    free(container->bytes);
    memset(container, 0, sizeof(*container));
}

static int read_container(libusb_device_handle *handle,
                          uint8_t bulk_in,
                          ptp_container *container,
                          unsigned timeout_ms) {
    memset(container, 0, sizeof(*container));
    uint8_t first[512];
    int received = 0;
    int result = libusb_bulk_transfer(handle, bulk_in, first, sizeof(first),
                                      &received, timeout_ms);
    if (result != 0) return result;
    if (received < 12) return LIBUSB_ERROR_IO;

    uint32_t length = get_u32(first);
    if (length < 12 || length > 128U * 1024U * 1024U) {
        fprintf(stderr, "invalid PTP container length %" PRIu32 "\n", length);
        return LIBUSB_ERROR_OVERFLOW;
    }
    if ((uint32_t)received > length) {
        fprintf(stderr, "coalesced PTP containers are not supported\n");
        return LIBUSB_ERROR_OVERFLOW;
    }

    uint8_t *bytes = malloc(length);
    if (!bytes) return LIBUSB_ERROR_NO_MEM;
    memcpy(bytes, first, (size_t)received);
    uint32_t offset = (uint32_t)received;
    while (offset < length) {
        int chunk = 0;
        uint32_t remaining = length - offset;
        int request = remaining > 1024U * 1024U
            ? 1024 * 1024 : (int)remaining;
        result = libusb_bulk_transfer(handle, bulk_in, bytes + offset, request,
                                      &chunk, timeout_ms);
        if (result != 0 || chunk <= 0) {
            free(bytes);
            return result != 0 ? result : LIBUSB_ERROR_IO;
        }
        offset += (uint32_t)chunk;
    }

    container->bytes = bytes;
    container->length = length;
    container->type = get_u16(bytes + 4);
    container->code = get_u16(bytes + 6);
    container->transaction = get_u32(bytes + 8);
    return 0;
}

static int transact(libusb_device_handle *handle,
                    uint8_t bulk_in,
                    uint8_t bulk_out,
                    uint16_t operation,
                    uint32_t transaction,
                    const uint32_t *parameters,
                    size_t parameter_count,
                    const uint8_t *out_data,
                    size_t out_data_length,
                    const char *data_path,
                    uint8_t **captured_data,
                    size_t *captured_data_length,
                    uint16_t *response_code) {
    if (captured_data) *captured_data = NULL;
    if (captured_data_length) *captured_data_length = 0;
    int result = send_command(handle, bulk_out, operation, transaction,
                              parameters, parameter_count);
    if (result != 0) return result;
    if (out_data != NULL) {
        result = send_data(handle, bulk_out, operation, transaction, out_data,
                           out_data_length);
        if (result != 0) return result;
        if (operation == PTP_OC_SEND_OBJECT && ptp_interrupt_in != 0) {
            /*
             * SXC completion publishes DRSPONSE.SXC through a PTP interrupt.
             * The camera's responder can wait for the event pipe to drain
             * before it completes the bulk response, unlike ordinary object
             * uploads. Keep polling briefly while the XML worker runs.
             */
            for (unsigned attempt = 0; attempt < 20; ++attempt) {
                uint8_t event[512];
                int event_length = 0;
                int event_result = libusb_interrupt_transfer(
                    handle, ptp_interrupt_in, event, sizeof(event),
                    &event_length, 250);
                if (event_result == 0 && event_length > 0) {
                    printf("interrupt event (%d bytes):", event_length);
                    for (int i = 0; i < event_length; ++i) {
                        printf(" %02x", event[i]);
                    }
                    printf("\n");
                    break;
                }
                if (event_result != LIBUSB_ERROR_TIMEOUT) {
                    fprintf(stderr, "interrupt endpoint: %s\n",
                            libusb_error_name(event_result));
                    break;
                }
            }
        }
    }

    unsigned stale_containers = 0;
    for (;;) {
        ptp_container container;
        unsigned response_timeout_ms =
            operation == PTP_OC_SEND_OBJECT ? 15000U : 5000U;
        result = read_container(handle, bulk_in, &container,
                                response_timeout_ms);
        if (result != 0) return result;
        printf("received type %u code 0x%04x transaction %" PRIu32
               " (%" PRIu32 " bytes)\n",
               container.type, container.code, container.transaction,
               container.length);

        if (container.transaction != transaction) {
            fprintf(stderr,
                    "discarding stale PTP transaction %" PRIu32
                    " while waiting for %" PRIu32 "\n",
                    container.transaction, transaction);
            free_container(&container);
            if (++stale_containers >= 8) return LIBUSB_ERROR_IO;
            continue;
        }
        if (container.type == PTP_CONTAINER_DATA) {
            size_t payload_length = container.length - 12;
            if (data_path) {
                FILE *output = fopen(data_path, "wb");
                if (!output) {
                    fprintf(stderr, "cannot open %s: %s\n",
                            data_path, strerror(errno));
                    free_container(&container);
                    return LIBUSB_ERROR_IO;
                }
                size_t written = payload_length == 0 ? 0 :
                    fwrite(container.bytes + 12, 1, payload_length, output);
                fclose(output);
                if (written != payload_length) {
                    fprintf(stderr, "short write to %s\n", data_path);
                    free_container(&container);
                    return LIBUSB_ERROR_IO;
                }
                printf("wrote %zu data byte(s) to %s\n",
                       payload_length, data_path);
            }
            if (captured_data && captured_data_length) {
                uint8_t *copy = NULL;
                if (payload_length != 0) {
                    copy = malloc(payload_length);
                    if (!copy) {
                        free_container(&container);
                        return LIBUSB_ERROR_NO_MEM;
                    }
                    memcpy(copy, container.bytes + 12, payload_length);
                }
                free(*captured_data);
                *captured_data = copy;
                *captured_data_length = payload_length;
            }
            free_container(&container);
            continue;
        }
        if (container.type != PTP_CONTAINER_RESPONSE) {
            fprintf(stderr, "unexpected PTP container type %u\n",
                    container.type);
            free_container(&container);
            return LIBUSB_ERROR_IO;
        }
        *response_code = container.code;
        free_container(&container);
        return 0;
    }
}

static size_t append_ptp_string(uint8_t *destination,
                                size_t capacity,
                                const char *text) {
    size_t length = strlen(text);
    if (length > 254 || capacity < 1 + 2 * (length + 1)) return 0;
    destination[0] = (uint8_t)(length + 1);
    for (size_t i = 0; i < length; ++i) {
        put_u16(destination + 1 + 2 * i, (uint8_t)text[i]);
    }
    put_u16(destination + 1 + 2 * length, 0);
    return 1 + 2 * (length + 1);
}

static size_t build_script_object_info(uint8_t *destination,
                                       size_t capacity,
                                       const char *filename) {
    if (capacity < 55) return 0;
    memset(destination, 0, capacity);
    put_u32(destination, XA5_INTERNAL_STORAGE_ID);
    put_u16(destination + 4, PTP_OFC_SCRIPT);
    size_t offset = 52;
    size_t filename_size =
        append_ptp_string(destination + offset, capacity - offset, filename);
    if (filename_size == 0) return 0;
    offset += filename_size;
    if (capacity - offset < 3) return 0;
    destination[offset++] = 0;  /* CaptureDate */
    destination[offset++] = 0;  /* ModificationDate */
    destination[offset++] = 0;  /* Keywords */
    return offset;
}

static int object_info_filename(const uint8_t *data,
                                size_t length,
                                char *filename,
                                size_t filename_capacity) {
    if (length < 53 || filename_capacity == 0) return -1;
    uint8_t units = data[52];
    if (units == 0) {
        filename[0] = '\0';
        return 0;
    }
    size_t required = 53U + 2U * units;
    if (required > length) return -1;
    size_t characters = units - 1U;
    if (characters >= filename_capacity) characters = filename_capacity - 1U;
    for (size_t i = 0; i < characters; ++i) {
        uint16_t character = get_u16(data + 53 + 2 * i);
        filename[i] = character <= 0x7f ? (char)character : '?';
    }
    filename[characters] = '\0';
    return 0;
}

static int list_objects(libusb_device_handle *handle,
                        uint8_t bulk_in,
                        uint8_t bulk_out,
                        uint32_t *transaction,
                        uint16_t *response,
                        uint16_t format_filter,
                        int fetch_sxc_responses) {
    uint32_t parameters[] = {0xffffffffU, format_filter, 0};
    uint8_t *handles_data = NULL;
    size_t handles_length = 0;
    int result = transact(handle, bulk_in, bulk_out,
                          PTP_OC_GET_OBJECT_HANDLES, (*transaction)++,
                          parameters, 3, NULL, 0, NULL,
                          &handles_data, &handles_length, response);
    if (result != 0 || *response != PTP_RC_OK) {
        fprintf(stderr,
                "GetObjectHandles(format 0x%04x) failed: transport %s, "
                "response 0x%04x\n",
                format_filter, libusb_error_name(result), *response);
        free(handles_data);
        return result != 0 ? result : LIBUSB_ERROR_OTHER;
    }
    if (handles_length < 4) {
        fprintf(stderr, "short object handle array\n");
        free(handles_data);
        return LIBUSB_ERROR_IO;
    }
    uint32_t count = get_u32(handles_data);
    if (count > (handles_length - 4) / 4) {
        fprintf(stderr, "invalid script handle count %" PRIu32 "\n", count);
        free(handles_data);
        return LIBUSB_ERROR_IO;
    }
    printf("camera exposes %" PRIu32
           " object(s) for format filter 0x%04x\n",
           count, format_filter);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t handle_value = get_u32(handles_data + 4 + 4 * i);
        uint32_t info_parameters[] = {handle_value};
        uint8_t *info_data = NULL;
        size_t info_length = 0;
        result = transact(handle, bulk_in, bulk_out, PTP_OC_GET_OBJECT_INFO,
                          (*transaction)++, info_parameters, 1, NULL, 0, NULL,
                          &info_data, &info_length, response);
        if (result != 0 || *response != PTP_RC_OK) {
            fprintf(stderr, "  handle 0x%08" PRIx32
                    ": GetObjectInfo failed (0x%04x)\n",
                    handle_value, *response);
            free(info_data);
            continue;
        }
        char filename[256];
        if (object_info_filename(info_data, info_length, filename,
                                 sizeof(filename)) != 0) {
            snprintf(filename, sizeof(filename), "<invalid ObjectInfo>");
        }
        uint16_t format = info_length >= 6 ? get_u16(info_data + 4) : 0;
        uint32_t size = info_length >= 12 ? get_u32(info_data + 8) : 0;
        printf("  handle 0x%08" PRIx32
               ": format 0x%04x, size %" PRIu32 ", %s\n",
               handle_value, format, size, filename);

        if (fetch_sxc_responses &&
            (strcmp(filename, "DDISCVRY.SXC") == 0 ||
             strcmp(filename, "DRSPONSE.SXC") == 0)) {
            char path[128];
            snprintf(path, sizeof(path), "/tmp/xa5-%s", filename);
            result = transact(handle, bulk_in, bulk_out, PTP_OC_GET_OBJECT,
                              (*transaction)++, info_parameters, 1,
                              NULL, 0, path, NULL, NULL, response);
            if (result != 0 || *response != PTP_RC_OK) {
                fprintf(stderr, "  GetObject(%s) failed (0x%04x)\n",
                        filename, *response);
            }
        }
        free(info_data);
    }
    free(handles_data);
    return 0;
}

static int valid_sxc_name(const char *name) {
    if (!name || strncmp(name, "Cam", 3) != 0) return 0;
    for (const unsigned char *p = (const unsigned char *)name; *p; ++p) {
        if (!((*p >= 'A' && *p <= 'Z') ||
              (*p >= 'a' && *p <= 'z') ||
              (*p >= '0' && *p <= '9') || *p == '_')) {
            return 0;
        }
    }
    return 1;
}

static int send_sxc_request(libusb_device_handle *handle,
                            uint8_t bulk_in,
                            uint8_t bulk_out,
                            uint32_t *transaction,
                            uint16_t *response,
                            const char *verb,
                            const char *name,
                            const char *value) {
    if (!valid_sxc_name(name)) {
        fprintf(stderr, "invalid SXC registry name: %s\n",
                name ? name : "(null)");
        return LIBUSB_ERROR_INVALID_PARAM;
    }
    const char *format_with_direct_value =
        "<?xml version=\"1.0\"?>"
        "<sxc xmlns=\"http://www.sanyo.co.jp/DSC/sxc/schema/\">"
        "<input><%s><%s>%s</%s></%s></input></sxc>";
    const char *format_with_value_node =
        "<?xml version=\"1.0\"?>"
        "<sxc xmlns=\"http://www.sanyo.co.jp/DSC/sxc/schema/\">"
        "<input><%s><%s><value>%s</value></%s></%s></input></sxc>";
    const char *format_without_value =
        "<?xml version=\"1.0\"?>"
        "<sxc xmlns=\"http://www.sanyo.co.jp/DSC/sxc/schema/\">"
        "<input><%s><%s/></%s></input></sxc>";
    const char *format_with_value =
        strcmp(verb, "Set") == 0
            ? format_with_value_node
            : format_with_direct_value;
    int required = value
        ? snprintf(NULL, 0, format_with_value, verb, name, value, name, verb)
        : snprintf(NULL, 0, format_without_value, verb, name, verb);
    if (required < 0) return LIBUSB_ERROR_OTHER;
    char *xml = malloc((size_t)required + 1);
    if (!xml) return LIBUSB_ERROR_NO_MEM;
    if (value) {
        (void)snprintf(xml, (size_t)required + 1, format_with_value,
                       verb, name, value, name, verb);
    } else {
        (void)snprintf(xml, (size_t)required + 1, format_without_value,
                       verb, name, verb);
    }

    uint8_t object_info[256];
    size_t object_info_length =
        build_script_object_info(object_info, sizeof(object_info),
                                 "HREQUEST.SXC");
    put_u32(object_info + 8, (uint32_t)required);
    uint32_t object_parameters[] = {XA5_INTERNAL_STORAGE_ID, 0};
    printf("sending SXC %s request for %s (%d XML bytes)\n",
           verb, name, required);
    printf("SXC request XML: %s\n", xml);
    int result = transact(handle, bulk_in, bulk_out, PTP_OC_SEND_OBJECT_INFO,
                          (*transaction)++, object_parameters, 2,
                          object_info, object_info_length, NULL,
                          NULL, NULL, response);
    if (result != 0 || *response != PTP_RC_OK) {
        fprintf(stderr,
                "HREQUEST.SXC SendObjectInfo failed: transport %s, "
                "response 0x%04x\n",
                libusb_error_name(result), *response);
        free(xml);
        return result != 0 ? result : LIBUSB_ERROR_OTHER;
    }
    result = transact(handle, bulk_in, bulk_out, PTP_OC_SEND_OBJECT,
                      (*transaction)++, NULL, 0,
                      (const uint8_t *)xml, (size_t)required, NULL,
                      NULL, NULL, response);
    free(xml);
    if (result != 0 || *response != PTP_RC_OK) {
        fprintf(stderr,
                "HREQUEST.SXC SendObject failed: transport %s, "
                "response 0x%04x\n",
                libusb_error_name(result), *response);
        return result != 0 ? result : LIBUSB_ERROR_OTHER;
    }
    return 0;
}

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s [--describe] [--movie] [--direct-movie] "
            "[--cancel-capture] "
            "[--terminate-movie-id N] "
            "[--get-video-out] [--set-video-out N] "
            "[--set-usb-mode N] [--set-force-mode N] "
            "[--preview FILE] [--delay-ms N] "
            "[--sxc-list] [--sxc-discover] [--sxc-get NAME] "
            "[--list-objects] "
            "[--sxc-set NAME HEXVALUE] "
            "[--sxc-execute NAME HEXVALUE --allow-sxc-execute] "
            "[--continue-after-sxc] [--skip-device-reset] [--hold]\n"
            "  --describe       print raw USB interface and endpoint layout\n"
            "  --movie          enter hidden Fuji movie-capture state\n"
            "  --direct-movie   enter movie state without open capture first\n"
            "  --cancel-capture\n"
            "                   send hidden 0x9030 before another requested\n"
            "                   capture operation, or by itself as cleanup\n"
            "  --terminate-movie-id N\n"
            "                   send hidden 0x9021 with a known initiating\n"
            "                   transaction id, then close the PTP session\n"
            "  --get-video-out  read Fuji VideoOutOnOff property 0xd168\n"
            "  --set-video-out N\n"
            "                   set 0xd168 using its camera-reported scalar\n"
            "                   datatype; may be chained before movie start\n"
            "  --set-usb-mode N set Fuji SetUSBMode 0xd15d as a u16\n"
            "  --set-force-mode N\n"
            "                   set Fuji ForceMode 0xd230 as a u16\n"
            "  --preview FILE   open capture, request 0x9022, save data phase\n"
            "  --delay-ms N     wait before requesting preview (default 1000)\n"
            "  --sxc-list       enumerate PTP script objects\n"
            "  --list-objects   enumerate all PTP objects and ObjectInfo\n"
            "  --sxc-discover   send the empty HDISCVRY.SXC discovery object\n"
            "  --sxc-get NAME   make a read-only SXC registry request\n"
            "  --sxc-set NAME HEXVALUE\n"
            "                   set one SXC registry value (1-8 hex digits)\n"
            "  --sxc-execute NAME HEXVALUE\n"
            "                   execute one SXC operation with a hex argument;\n"
            "                   requires the explicit execution interlock below\n"
            "  --allow-sxc-execute\n"
            "                   acknowledge that Execute can start calibration\n"
            "                   hardware, block the camera, or disrupt USB\n"
            "  --continue-after-sxc\n"
            "                   keep this PTP session open and run the requested\n"
            "                   capture operation immediately after SXC\n"
            "  --skip-device-reset\n"
            "                   preserve camera RAM across a coordinated,\n"
            "                   multi-session SXC transaction; unsafe if an\n"
            "                   unrelated PTP owner left a stale session\n"
            "  --hold           keep open-capture active until Control-C\n",
            program);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);

    const char *preview_path = NULL;
    unsigned delay_ms = 1000;
    int hold = 0;
    int movie = 0;
    int direct_movie = 0;
    int cancel_capture = 0;
    int terminate_movie = 0;
    uint32_t terminate_movie_id = 0;
    int get_video_out = 0;
    int set_video_out = 0;
    uint64_t video_out_value = 0;
    int set_usb_mode = 0;
    uint16_t usb_mode_value = 0;
    int set_force_mode = 0;
    uint16_t force_mode_value = 0;
    int sxc_list = 0;
    int list_all_objects = 0;
    int sxc_discover = 0;
    const char *sxc_get_name = NULL;
    const char *sxc_set_name = NULL;
    const char *sxc_set_value = NULL;
    const char *sxc_execute_name = NULL;
    const char *sxc_execute_value = NULL;
    int allow_sxc_execute = 0;
    int continue_after_sxc = 0;
    int skip_device_reset = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--describe") == 0) {
            continue;
        } else if (strcmp(argv[i], "--movie") == 0) {
            movie = 1;
        } else if (strcmp(argv[i], "--direct-movie") == 0) {
            movie = 1;
            direct_movie = 1;
        } else if (strcmp(argv[i], "--cancel-capture") == 0) {
            cancel_capture = 1;
        } else if (strcmp(argv[i], "--terminate-movie-id") == 0 &&
                   i + 1 < argc) {
            char *end = NULL;
            unsigned long parsed = strtoul(argv[++i], &end, 0);
            if (!end || *end != '\0' || parsed > UINT32_MAX) {
                usage(argv[0]);
                return 2;
            }
            terminate_movie = 1;
            terminate_movie_id = (uint32_t)parsed;
        } else if (strcmp(argv[i], "--get-video-out") == 0) {
            get_video_out = 1;
        } else if (strcmp(argv[i], "--set-video-out") == 0 &&
                   i + 1 < argc) {
            char *end = NULL;
            errno = 0;
            unsigned long long parsed = strtoull(argv[++i], &end, 0);
            if (errno != 0 || !end || *end != '\0') {
                usage(argv[0]);
                return 2;
            }
            get_video_out = 1;
            set_video_out = 1;
            video_out_value = (uint64_t)parsed;
        } else if ((strcmp(argv[i], "--set-usb-mode") == 0 ||
                    strcmp(argv[i], "--set-force-mode") == 0) &&
                   i + 1 < argc) {
            int is_usb_mode = strcmp(argv[i], "--set-usb-mode") == 0;
            char *end = NULL;
            errno = 0;
            unsigned long parsed = strtoul(argv[++i], &end, 0);
            if (errno != 0 || !end || *end != '\0' ||
                parsed > UINT16_MAX) {
                usage(argv[0]);
                return 2;
            }
            if (is_usb_mode) {
                set_usb_mode = 1;
                usb_mode_value = (uint16_t)parsed;
            } else {
                set_force_mode = 1;
                force_mode_value = (uint16_t)parsed;
            }
        } else if (strcmp(argv[i], "--preview") == 0 && i + 1 < argc) {
            preview_path = argv[++i];
        } else if (strcmp(argv[i], "--delay-ms") == 0 && i + 1 < argc) {
            char *end = NULL;
            unsigned long parsed = strtoul(argv[++i], &end, 0);
            if (!end || *end != '\0' || parsed > 600000) {
                usage(argv[0]);
                return 2;
            }
            delay_ms = (unsigned)parsed;
        } else if (strcmp(argv[i], "--sxc-list") == 0) {
            sxc_list = 1;
        } else if (strcmp(argv[i], "--list-objects") == 0) {
            list_all_objects = 1;
        } else if (strcmp(argv[i], "--sxc-discover") == 0) {
            sxc_list = 1;
            sxc_discover = 1;
        } else if (strcmp(argv[i], "--sxc-get") == 0 && i + 1 < argc) {
            sxc_list = 1;
            sxc_discover = 1;
            sxc_get_name = argv[++i];
        } else if (strcmp(argv[i], "--sxc-set") == 0 && i + 2 < argc) {
            sxc_list = 1;
            sxc_discover = 1;
            sxc_set_name = argv[++i];
            sxc_set_value = argv[++i];
            size_t value_length = strlen(sxc_set_value);
            if (value_length < 1 || value_length > 8) {
                usage(argv[0]);
                return 2;
            }
            for (const unsigned char *p =
                     (const unsigned char *)sxc_set_value; *p; ++p) {
                if (!((*p >= '0' && *p <= '9') ||
                      (*p >= 'a' && *p <= 'f') ||
                      (*p >= 'A' && *p <= 'F'))) {
                    usage(argv[0]);
                    return 2;
                }
            }
        } else if (strcmp(argv[i], "--sxc-execute") == 0 &&
                   i + 2 < argc) {
            sxc_list = 1;
            sxc_discover = 1;
            sxc_execute_name = argv[++i];
            sxc_execute_value = argv[++i];
            size_t value_length = strlen(sxc_execute_value);
            if (value_length < 1 || value_length > 8) {
                usage(argv[0]);
                return 2;
            }
            for (const unsigned char *p =
                     (const unsigned char *)sxc_execute_value; *p; ++p) {
                if (!((*p >= '0' && *p <= '9') ||
                      (*p >= 'a' && *p <= 'f') ||
                      (*p >= 'A' && *p <= 'F'))) {
                    usage(argv[0]);
                    return 2;
                }
            }
        } else if (strcmp(argv[i], "--allow-sxc-execute") == 0) {
            allow_sxc_execute = 1;
        } else if (strcmp(argv[i], "--continue-after-sxc") == 0) {
            continue_after_sxc = 1;
        } else if (strcmp(argv[i], "--skip-device-reset") == 0) {
            skip_device_reset = 1;
        } else if (strcmp(argv[i], "--hold") == 0) {
            hold = 1;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (sxc_execute_name && !allow_sxc_execute) {
        fprintf(stderr,
                "refusing SXC Execute without --allow-sxc-execute; "
                "Execute is a state-changing factory operation\n");
        return 2;
    }
    if (continue_after_sxc && !movie && !preview_path && !hold &&
        !cancel_capture && !terminate_movie) {
        fprintf(stderr,
                "--continue-after-sxc requires a following capture or "
                "cleanup operation\n");
        return 2;
    }

    libusb_context *context = NULL;
    libusb_device_handle *handle = NULL;
    struct libusb_config_descriptor *config = NULL;
    int claimed = 0;
    int interface_number = -1;
    uint8_t bulk_in = 0, bulk_out = 0, interrupt_in = 0;
    int status = 1;

    int result = libusb_init(&context);
    if (result != 0) {
        fprintf(stderr, "libusb_init: %s\n", libusb_error_name(result));
        return 1;
    }
    handle = libusb_open_device_with_vid_pid(context, XA5_VENDOR_ID,
                                              XA5_PRODUCT_ID);
    if (!handle) {
        fprintf(stderr, "Fujifilm 04cb:02d5 not available\n");
        goto cleanup;
    }
    libusb_device *device = libusb_get_device(handle);
    result = libusb_get_active_config_descriptor(device, &config);
    if (result != 0) {
        fprintf(stderr, "configuration descriptor: %s\n",
                libusb_error_name(result));
        goto cleanup;
    }
    if (!find_ptp_interface(config, &interface_number, &bulk_in, &bulk_out,
                            &interrupt_in)) {
        fprintf(stderr, "no complete PTP interface found\n");
        goto cleanup;
    }
    ptp_interrupt_in = interrupt_in;
    printf("PTP transport: interface %d, bulk-out 0x%02x, bulk-in 0x%02x",
           interface_number, bulk_out, bulk_in);
    if (interrupt_in) printf(", interrupt-in 0x%02x", interrupt_in);
    printf("\n");

    if (!preview_path && !hold && !movie && !cancel_capture &&
        !terminate_movie && !get_video_out &&
        !set_usb_mode && !set_force_mode && !sxc_list &&
        !list_all_objects &&
        !sxc_get_name && !sxc_set_name && !sxc_execute_name) {
        status = 0;
        goto cleanup;
    }

    result = libusb_claim_interface(handle, interface_number);
    if (result != 0) {
        fprintf(stderr, "claim interface %d: %s\n", interface_number,
                libusb_error_name(result));
        fprintf(stderr, "Close camera applications and stop ptpcamerad/icdd "
                        "before the raw transport test.\n");
        goto cleanup;
    }
    claimed = 1;

    /*
     * Apple may have been killed while a PTP session or transfer was open.
     * Use the Still Image class reset request, then clear both bulk pipes,
     * before establishing our own raw session.
     */
    if (skip_device_reset) {
        printf("preserving camera RAM: skipped PTP class-specific reset\n");
    } else {
        result = libusb_control_transfer(
            handle,
            LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_CLASS |
                LIBUSB_RECIPIENT_INTERFACE,
            PTP_USB_REQUEST_DEVICE_RESET, 0, (uint16_t)interface_number,
            NULL, 0, 3000);
        if (result < 0) {
            fprintf(stderr, "PTP class reset: %s\n",
                    libusb_error_name(result));
        } else {
            printf("issued PTP class-specific device reset\n");
        }
        usleep(250000);
    }
    (void)libusb_clear_halt(handle, bulk_out);
    (void)libusb_clear_halt(handle, bulk_in);

    uint32_t transaction = 1;
    uint16_t response = 0;
    uint32_t session_parameters[] = {1};
    result = transact(handle, bulk_in, bulk_out, PTP_OC_OPEN_SESSION,
                      transaction++, session_parameters, 1, NULL, 0, NULL,
                      NULL, NULL, &response);
    if (result == 0 && response == PTP_RC_SESSION_ALREADY_OPEN) {
        /*
         * ptpcamerad can leave the camera-side session alive when forcibly
         * detached. Close that stale session over the newly claimed raw
         * transport, then retry with a fresh session id.
         */
        fprintf(stderr,
                "camera reports stale session state; closing and retrying\n");
        (void)transact(handle, bulk_in, bulk_out, PTP_OC_CLOSE_SESSION,
                       transaction++, NULL, 0, NULL, 0, NULL,
                       NULL, NULL, &response);
        session_parameters[0] = 2;
        result = transact(handle, bulk_in, bulk_out, PTP_OC_OPEN_SESSION,
                          transaction++, session_parameters, 1,
                          NULL, 0, NULL, NULL, NULL, &response);
    }
    if (result != 0 || response != PTP_RC_OK) {
        fprintf(stderr, "OpenSession failed: transport %s, response 0x%04x\n",
                libusb_error_name(result), response);
        goto cleanup;
    }

    if (sxc_list) {
        printf("SXC script objects before discovery:\n");
        result = list_objects(handle, bulk_in, bulk_out, &transaction,
                              &response, PTP_OFC_SCRIPT, 1);
        if (result != 0) goto close_session;
        if (!sxc_discover) goto close_session;
    }

    if (list_all_objects) {
        printf("All PTP objects:\n");
        result = list_objects(handle, bulk_in, bulk_out, &transaction,
                              &response, 0, 0);
        goto close_session;
    }

    if (sxc_discover) {
        uint8_t object_info[256];
        size_t object_info_length =
            build_script_object_info(object_info, sizeof(object_info),
                                     "HDISCVRY.SXC");
        uint32_t discovery_parameters[] = {XA5_INTERNAL_STORAGE_ID, 0};
        printf("sending benign SXC discovery ObjectInfo\n");
        result = transact(handle, bulk_in, bulk_out, PTP_OC_SEND_OBJECT_INFO,
                          transaction++, discovery_parameters, 2,
                          object_info, object_info_length, NULL,
                          NULL, NULL, &response);
        printf("HDISCVRY.SXC response: 0x%04x\n", response);
        if (result != 0 || response != PTP_RC_OK) goto close_session;
        if (sxc_get_name) {
            result = send_sxc_request(handle, bulk_in, bulk_out, &transaction,
                                      &response, "Get", sxc_get_name, NULL);
            if (result != 0) goto close_session;
            usleep(500000);
        } else if (sxc_set_name) {
            result = send_sxc_request(handle, bulk_in, bulk_out, &transaction,
                                      &response, "Set", sxc_set_name,
                                      sxc_set_value);
            if (result != 0) goto close_session;
            usleep(500000);
        } else if (sxc_execute_name) {
            result = send_sxc_request(handle, bulk_in, bulk_out, &transaction,
                                      &response, "Execute", sxc_execute_name,
                                      sxc_execute_value);
            if (result != 0) goto close_session;
            /*
             * Execute is queued onto the camera's factory worker.  The PTP
             * SendObject response only acknowledges that queueing.  On the
             * X-A5, DRSPONSE.SXC is not readable in this PTP session even
             * after a long delay; it becomes readable immediately after a
             * clean close/reopen.  Do not wedge the bulk endpoint by asking
             * for the half-published ObjectInfo here.  The next SXC command
             * lists and saves the deferred response before sending its own.
             */
            usleep(500000);
            printf("SXC Execute response deferred until the next session\n");
            if (!preview_path && !continue_after_sxc) goto close_session;
        }
        if (!sxc_execute_name) {
            usleep(250000);
            printf("SXC script objects after discovery:\n");
            result = list_objects(handle, bulk_in, bulk_out, &transaction,
                                  &response, PTP_OFC_SCRIPT, 1);
            if (result != 0 || !continue_after_sxc) goto close_session;
        }
        printf("continuing in the same PTP session after SXC request\n");
    }

    if (set_usb_mode || set_force_mode) {
        uint16_t property_code = set_usb_mode
            ? PTP_DPC_FUJI_SET_USB_MODE : PTP_DPC_FUJI_FORCE_MODE;
        uint16_t property_value = set_usb_mode
            ? usb_mode_value : force_mode_value;
        const char *property_name = set_usb_mode
            ? "SetUSBMode" : "ForceMode";
        uint32_t property_parameters[] = {property_code};
        uint8_t value_bytes[2];
        put_u16(value_bytes, property_value);
        result = transact(handle, bulk_in, bulk_out,
                          PTP_OC_SET_DEVICE_PROP_VALUE, transaction++,
                          property_parameters, 1,
                          value_bytes, sizeof(value_bytes), NULL,
                          NULL, NULL, &response);
        printf("Set %s(0x%04x)=0x%04x response: 0x%04x\n",
               property_name, property_code, property_value, response);
        if (result != 0 || response != PTP_RC_OK) {
            if (result == 0) result = LIBUSB_ERROR_NOT_SUPPORTED;
            goto close_session;
        }
        if (!movie && !preview_path && !hold && !cancel_capture &&
            !get_video_out) {
            goto close_session;
        }
    }

    if (get_video_out) {
        uint32_t property_parameters[] = {
            PTP_DPC_FUJI_VIDEO_OUT_ON_OFF
        };
        uint8_t *descriptor = NULL;
        size_t descriptor_length = 0;
        result = transact(handle, bulk_in, bulk_out,
                          PTP_OC_GET_DEVICE_PROP_DESC, transaction++,
                          property_parameters, 1, NULL, 0, NULL,
                          &descriptor, &descriptor_length, &response);
        if (result != 0 || response != PTP_RC_OK) {
            fprintf(stderr,
                    "VideoOutOnOff descriptor failed: transport %s, "
                    "response 0x%04x\n",
                    libusb_error_name(result), response);
            free(descriptor);
            uint8_t *raw_value = NULL;
            size_t raw_value_length = 0;
            result = transact(handle, bulk_in, bulk_out,
                              PTP_OC_GET_DEVICE_PROP_VALUE, transaction++,
                              property_parameters, 1, NULL, 0, NULL,
                              &raw_value, &raw_value_length, &response);
            printf("Direct VideoOutOnOff value response: 0x%04x\n",
                   response);
            if (raw_value_length != 0) {
                print_hex_bytes("Direct VideoOutOnOff value",
                                raw_value, raw_value_length);
            }
            free(raw_value);
            if (!set_video_out) {
                if (result == 0) result = LIBUSB_ERROR_NOT_SUPPORTED;
                goto close_session;
            }
            if (video_out_value > UINT16_MAX) {
                fprintf(stderr,
                        "legacy VideoOutOnOff fallback requires a u16 value\n");
                result = LIBUSB_ERROR_INVALID_PARAM;
                goto close_session;
            }
            uint8_t legacy_value[2];
            put_u16(legacy_value, (uint16_t)video_out_value);
            printf("trying legacy Fuji u16 VideoOutOnOff setter fallback\n");
            result = transact(handle, bulk_in, bulk_out,
                              PTP_OC_SET_DEVICE_PROP_VALUE, transaction++,
                              property_parameters, 1,
                              legacy_value, sizeof(legacy_value), NULL,
                              NULL, NULL, &response);
            printf("Set VideoOutOnOff=0x%" PRIx64
                   " u16 fallback response: 0x%04x\n",
                   video_out_value, response);
            if (result != 0 || response != PTP_RC_OK) {
                if (result == 0) result = LIBUSB_ERROR_NOT_SUPPORTED;
                goto close_session;
            }
            goto video_out_complete;
        }
        print_hex_bytes("VideoOutOnOff descriptor", descriptor,
                        descriptor_length);
        if (descriptor_length < 5 ||
            get_u16(descriptor) != PTP_DPC_FUJI_VIDEO_OUT_ON_OFF) {
            fprintf(stderr, "malformed VideoOutOnOff descriptor\n");
            free(descriptor);
            result = LIBUSB_ERROR_IO;
            goto close_session;
        }
        uint16_t property_type = get_u16(descriptor + 2);
        uint8_t property_get_set = descriptor[4];
        size_t property_size = ptp_scalar_size(property_type);
        printf("VideoOutOnOff datatype 0x%04x, %s, scalar size %zu\n",
               property_type,
               property_get_set ? "get/set" : "get-only",
               property_size);
        free(descriptor);
        if (property_size == 0) {
            fprintf(stderr,
                    "VideoOutOnOff is not a supported scalar datatype\n");
            result = LIBUSB_ERROR_NOT_SUPPORTED;
            goto close_session;
        }

        uint8_t *property_data = NULL;
        size_t property_data_length = 0;
        result = transact(handle, bulk_in, bulk_out,
                          PTP_OC_GET_DEVICE_PROP_VALUE, transaction++,
                          property_parameters, 1, NULL, 0, NULL,
                          &property_data, &property_data_length, &response);
        if (result != 0 || response != PTP_RC_OK ||
            property_data_length != property_size) {
            fprintf(stderr,
                    "VideoOutOnOff value failed: transport %s, "
                    "response 0x%04x, size %zu\n",
                    libusb_error_name(result), response,
                    property_data_length);
            free(property_data);
            if (result == 0) result = LIBUSB_ERROR_IO;
            goto close_session;
        }
        printf("VideoOutOnOff current value: 0x%" PRIx64 "\n",
               get_scalar(property_data, property_size));
        free(property_data);

        if (set_video_out) {
            if (!property_get_set) {
                fprintf(stderr, "VideoOutOnOff is camera-reported get-only\n");
                result = LIBUSB_ERROR_ACCESS;
                goto close_session;
            }
            if (property_size < sizeof(video_out_value) &&
                video_out_value >= (UINT64_C(1) << (property_size * 8))) {
                fprintf(stderr,
                        "VideoOutOnOff value does not fit datatype 0x%04x\n",
                        property_type);
                result = LIBUSB_ERROR_INVALID_PARAM;
                goto close_session;
            }
            uint8_t value_bytes[8] = {0};
            put_scalar(value_bytes, property_size, video_out_value);
            result = transact(handle, bulk_in, bulk_out,
                              PTP_OC_SET_DEVICE_PROP_VALUE, transaction++,
                              property_parameters, 1,
                              value_bytes, property_size, NULL,
                              NULL, NULL, &response);
            printf("Set VideoOutOnOff=0x%" PRIx64
                   " response: 0x%04x\n",
                   video_out_value, response);
            if (result != 0 || response != PTP_RC_OK) {
                if (result == 0) result = LIBUSB_ERROR_OTHER;
                goto close_session;
            }
        }
video_out_complete:
        if (!movie && !preview_path && !hold && !cancel_capture) {
            goto close_session;
        }
    }

    if (cancel_capture) {
        result = transact(handle, bulk_in, bulk_out,
                          PTP_OC_CANCEL_INITIATE_CAPTURE, transaction++,
                          NULL, 0, NULL, 0, NULL,
                          NULL, NULL, &response);
        if (result != 0) {
            fprintf(stderr, "CancelInitiateCapture transport: %s\n",
                    libusb_error_name(result));
            goto close_session;
        }
        printf("CancelInitiateCapture response: 0x%04x\n", response);
        if (!movie && !preview_path && !hold) goto close_session;
    }

    if (terminate_movie) {
        uint32_t terminate_parameters[] = {terminate_movie_id};
        result = transact(handle, bulk_in, bulk_out,
                          PTP_OC_TERMINATE_MOVIE_CAPTURE, transaction++,
                          terminate_parameters, 1, NULL, 0, NULL,
                          NULL, NULL, &response);
        if (result != 0) {
            fprintf(stderr, "TerminateMovieCapture transport: %s\n",
                    libusb_error_name(result));
        } else {
            printf("TerminateMovieCapture response: 0x%04x\n", response);
        }
        goto close_session;
    }

    uint32_t capture_transaction = 0;
    if (!direct_movie) {
        uint32_t capture_parameters[] = {0, 0};
        capture_transaction = transaction;
        result = transact(handle, bulk_in, bulk_out,
                          PTP_OC_INITIATE_OPEN_CAPTURE, transaction++,
                          capture_parameters, 2, NULL, 0, NULL,
                          NULL, NULL, &response);
        if (result != 0 || response != PTP_RC_OK) {
            fprintf(stderr,
                    "InitiateOpenCapture failed: transport %s, "
                    "response 0x%04x\n",
                    libusb_error_name(result), response);
            goto close_session;
        }
    }

    uint32_t movie_transaction = 0;
    if (movie) {
        uint32_t movie_parameters[] = {0};
        movie_transaction = transaction;
        result = transact(handle, bulk_in, bulk_out,
                          PTP_OC_INITIATE_MOVIE_CAPTURE, transaction++,
                          movie_parameters, 1, NULL, 0, NULL,
                          NULL, NULL, &response);
        if (result != 0 || response != PTP_RC_OK) {
            fprintf(stderr,
                    "InitiateMovieCapture failed: transport %s, "
                    "response 0x%04x\n",
                    libusb_error_name(result), response);
            movie_transaction = 0;
        }
    }

    if (preview_path) {
        usleep((useconds_t)delay_ms * 1000);
        result = transact(handle, bulk_in, bulk_out,
                          PTP_OC_GET_CAPTURE_PREVIEW, transaction++,
                          NULL, 0, NULL, 0, preview_path,
                          NULL, NULL, &response);
        if (result != 0) {
            fprintf(stderr, "GetCapturePreview transport: %s\n",
                    libusb_error_name(result));
        } else {
            printf("GetCapturePreview response: 0x%04x\n", response);
        }
    }

    if (hold) {
        signal(SIGINT, on_signal);
        signal(SIGTERM, on_signal);
        printf("holding raw capture state");
        if (capture_transaction) {
            printf(" (open transaction %" PRIu32 ")", capture_transaction);
        }
        if (movie_transaction) {
            printf(" (movie transaction %" PRIu32 ")", movie_transaction);
        }
        printf("; press Control-C to terminate\n");
        while (!interrupted) {
            if (interrupt_in) {
                uint8_t event[512];
                int event_length = 0;
                int event_result = libusb_interrupt_transfer(
                    handle, interrupt_in, event, sizeof(event), &event_length,
                    250);
                if (event_result == 0 && event_length > 0) {
                    printf("interrupt event (%d bytes):", event_length);
                    for (int i = 0; i < event_length; ++i) {
                        printf(" %02x", event[i]);
                    }
                    printf("\n");
                } else if (event_result != LIBUSB_ERROR_TIMEOUT) {
                    fprintf(stderr, "interrupt endpoint: %s\n",
                            libusb_error_name(event_result));
                    break;
                }
            } else {
                usleep(250000);
            }
        }
    }

    if (movie_transaction != 0) {
        uint32_t movie_terminate_parameters[] = {movie_transaction};
        (void)transact(handle, bulk_in, bulk_out,
                       PTP_OC_TERMINATE_MOVIE_CAPTURE, transaction++,
                       movie_terminate_parameters, 1, NULL, 0, NULL,
                       NULL, NULL, &response);
    }

    if (capture_transaction != 0) {
        uint32_t terminate_parameters[] = {capture_transaction};
        (void)transact(handle, bulk_in, bulk_out,
                       PTP_OC_TERMINATE_OPEN_CAPTURE, transaction++,
                       terminate_parameters, 1, NULL, 0, NULL,
                       NULL, NULL, &response);
    }

close_session:
    (void)transact(handle, bulk_in, bulk_out, PTP_OC_CLOSE_SESSION,
                   transaction++, NULL, 0, NULL, 0, NULL,
                   NULL, NULL, &response);
    status = result == 0 ? 0 : 1;

cleanup:
    if (claimed) libusb_release_interface(handle, interface_number);
    if (config) libusb_free_config_descriptor(config);
    if (handle) libusb_close(handle);
    if (context) libusb_exit(context);
    return status;
}
