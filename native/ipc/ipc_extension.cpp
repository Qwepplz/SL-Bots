#include "protocol.hpp"

#include <windows.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

using sl_bots::ipc::ProtocolHeaderV1;

constexpr std::uint32_t kRingMagic = 0x534C4252u;
constexpr std::uint32_t kRingVersion = 1;
constexpr std::uint32_t kDefaultCapacity = 64;
constexpr std::uint32_t kMaxCapacity = 1024;
constexpr int kOpenError = -1;
constexpr int kFullError = -2;
constexpr int kAbortedError = -3;
constexpr int kProtocolError = -4;

#pragma pack(push, 1)
struct RingHeaderV1 {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t capacity;
    std::uint32_t packet_size;
    volatile LONG64 write_index;
    volatile LONG64 read_index;
    volatile LONG epoch;
    volatile LONG aborted;
    volatile LONG64 heartbeat;
};
#pragma pack(pop)

static_assert(sizeof(RingHeaderV1) == 48);

std::uint64_t ReadCounter(const volatile LONG64* value) {
    return static_cast<std::uint64_t>(InterlockedCompareExchange64(
        const_cast<volatile LONG64*>(value), 0, 0));
}

std::uint32_t Crc32(const std::uint8_t* data, std::size_t size) {
    std::uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t index = 0; index < size; ++index) {
        crc ^= data[index];
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = static_cast<std::uint32_t>(-
                static_cast<std::int32_t>(crc & 1u));
            crc = (crc >> 1u) ^ (0xEDB88320u & mask);
        }
    }
    return crc ^ 0xFFFFFFFFu;
}

bool IsHeaderValid(
    const ProtocolHeaderV1& header,
    std::uint16_t expectedPacketSize) {
    return std::memcmp(header.magic, sl_bots::ipc::kProtocolMagic, sizeof(header.magic)) == 0 &&
        header.schema_version == sl_bots::ipc::kSchemaVersion &&
        header.record_size == expectedPacketSize &&
        header.bot_count <= sl_bots::ipc::kMaxBots &&
        header.reserved == 0;
}

bool IsPacketValid(
    const std::uint8_t* packet,
    std::size_t packetSize,
    std::uint16_t expectedPacketSize) {
    if (packet == nullptr || packetSize != expectedPacketSize || packetSize < sizeof(ProtocolHeaderV1)) {
        return false;
    }
    const auto* header = reinterpret_cast<const ProtocolHeaderV1*>(packet);
    if (!IsHeaderValid(*header, expectedPacketSize)) {
        return false;
    }
    ProtocolHeaderV1 headerWithoutCrc = *header;
    const std::uint32_t checksum = headerWithoutCrc.crc32;
    headerWithoutCrc.crc32 = 0;
    std::vector<std::uint8_t> crcInput(packetSize);
    std::memcpy(crcInput.data(), &headerWithoutCrc, sizeof(headerWithoutCrc));
    std::memcpy(
        crcInput.data() + sizeof(headerWithoutCrc),
        packet + sizeof(headerWithoutCrc),
        packetSize - sizeof(headerWithoutCrc));
    return checksum == Crc32(crcInput.data(), crcInput.size());
}

std::wstring Utf8ToWide(const char* value) {
    if (value == nullptr || value[0] == '\0') {
        return {};
    }
    const int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value, -1, nullptr, 0);
    if (length <= 0) {
        return {};
    }
    std::wstring result(static_cast<std::size_t>(length), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value, -1, result.data(), length) <= 0) {
        return {};
    }
    result.pop_back();
    return result;
}

class SharedRing final {
public:
    SharedRing() = default;
    SharedRing(const SharedRing&) = delete;
    SharedRing& operator=(const SharedRing&) = delete;

    ~SharedRing() { Close(); }

    bool Open(
        const std::wstring& name,
        std::uint32_t epoch,
        std::uint32_t capacity,
        std::uint32_t packetSize) {
        Close();
        if (name.empty() || capacity == 0 || capacity > kMaxCapacity || packetSize == 0) {
            return false;
        }
        const std::uint64_t mappingSize = sizeof(RingHeaderV1) +
            static_cast<std::uint64_t>(capacity) * packetSize;
        if (mappingSize > std::numeric_limits<DWORD>::max()) {
            return false;
        }
        mapping_ = CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            nullptr,
            PAGE_READWRITE,
            0,
            static_cast<DWORD>(mappingSize),
            name.c_str());
        if (mapping_ == nullptr) {
            return false;
        }
        const bool created = GetLastError() != ERROR_ALREADY_EXISTS;
        view_ = static_cast<std::uint8_t*>(MapViewOfFile(mapping_, FILE_MAP_ALL_ACCESS, 0, 0, 0));
        if (view_ == nullptr) {
            Close();
            return false;
        }
        header_ = reinterpret_cast<RingHeaderV1*>(view_);
        if (created) {
            std::memset(view_, 0, static_cast<std::size_t>(mappingSize));
            header_->magic = kRingMagic;
            header_->version = kRingVersion;
            header_->capacity = capacity;
            header_->packet_size = packetSize;
            header_->epoch = static_cast<LONG>(epoch);
            header_->heartbeat = static_cast<LONG64>(GetTickCount64());
        } else if (header_->magic != kRingMagic ||
                   header_->version != kRingVersion ||
                   header_->capacity != capacity ||
                   header_->packet_size != packetSize) {
            Close();
            return false;
        }
        if (static_cast<std::uint32_t>(header_->epoch) != epoch) {
            Close();
            return false;
        }
        return true;
    }

    void Close() {
        header_ = nullptr;
        if (view_ != nullptr) {
            UnmapViewOfFile(view_);
            view_ = nullptr;
        }
        if (mapping_ != nullptr) {
            CloseHandle(mapping_);
            mapping_ = nullptr;
        }
    }

    int Write(const void* packet, std::size_t packetSize) {
        if (!Ready() || packet == nullptr || packetSize != header_->packet_size) {
            return kOpenError;
        }
        if (header_->aborted != 0) {
            return kAbortedError;
        }
        const std::uint64_t writeIndex = ReadCounter(&header_->write_index);
        const std::uint64_t readIndex = ReadCounter(&header_->read_index);
        if (writeIndex - readIndex >= header_->capacity) {
            return kFullError;
        }
        std::uint8_t* slot = Slot(writeIndex);
        std::memcpy(slot, packet, packetSize);
        MemoryBarrier();
        InterlockedExchange64(&header_->write_index, static_cast<LONG64>(writeIndex + 1));
        InterlockedExchange64(&header_->heartbeat, static_cast<LONG64>(GetTickCount64()));
        return 1;
    }

    int Read(void* packet, std::size_t packetSize) {
        if (!Ready() || packet == nullptr || packetSize != header_->packet_size) {
            return kOpenError;
        }
        if (header_->aborted != 0) {
            return kAbortedError;
        }
        const std::uint64_t writeIndex = ReadCounter(&header_->write_index);
        const std::uint64_t readIndex = ReadCounter(&header_->read_index);
        if (writeIndex == readIndex) {
            return 0;
        }
        MemoryBarrier();
        std::memcpy(packet, Slot(readIndex), packetSize);
        InterlockedExchange64(&header_->read_index, static_cast<LONG64>(readIndex + 1));
        InterlockedExchange64(&header_->heartbeat, static_cast<LONG64>(GetTickCount64()));
        return 1;
    }

    void Abort() {
        if (Ready()) {
            InterlockedExchange(&header_->aborted, 1);
            InterlockedExchange64(&header_->heartbeat, static_cast<LONG64>(GetTickCount64()));
        }
    }

    bool SwitchEpoch(std::uint32_t epoch) {
        if (!Ready()) {
            return false;
        }
        InterlockedExchange64(&header_->write_index, 0);
        InterlockedExchange64(&header_->read_index, 0);
        InterlockedExchange(&header_->epoch, static_cast<LONG>(epoch));
        InterlockedExchange(&header_->aborted, 0);
        InterlockedExchange64(&header_->heartbeat, static_cast<LONG64>(GetTickCount64()));
        return true;
    }

    std::uint64_t Heartbeat() const {
        return Ready() ? ReadCounter(&header_->heartbeat) : 0;
    }

private:
    bool Ready() const { return mapping_ != nullptr && view_ != nullptr && header_ != nullptr; }

    std::uint8_t* Slot(std::uint64_t index) const {
        const std::uint64_t offset = sizeof(RingHeaderV1) +
            (index % header_->capacity) * header_->packet_size;
        return view_ + offset;
    }

    HANDLE mapping_ = nullptr;
    std::uint8_t* view_ = nullptr;
    RingHeaderV1* header_ = nullptr;
};

class Transport final {
public:
    int Open(const char* name, std::uint32_t epoch, std::uint32_t capacity) {
        const std::wstring base = Utf8ToWide(name);
        if (base.empty() || capacity == 0 || capacity > kMaxCapacity) {
            return kOpenError;
        }
        const std::wstring observationName = base + L"_obs_v1";
        const std::wstring actionName = base + L"_act_v1";
        if (!observations_.Open(
                observationName,
                epoch,
                capacity,
                sl_bots::ipc::kObservationPacketSize) ||
            !actions_.Open(actionName, epoch, capacity, sl_bots::ipc::kActionPacketSize)) {
            Close();
            return kOpenError;
        }
        return 1;
    }

    void Close() {
        observations_.Close();
        actions_.Close();
    }

    int PublishObservation(const void* packet, std::size_t packetSize) {
        if (!IsPacketValid(
                static_cast<const std::uint8_t*>(packet),
                packetSize,
                sl_bots::ipc::kObservationPacketSize)) {
            return kProtocolError;
        }
        return observations_.Write(packet, packetSize);
    }

    int TryReadAction(void* packet, std::size_t packetSize) {
        if (packet == nullptr || packetSize != sl_bots::ipc::kActionPacketSize) {
            return kProtocolError;
        }
        std::array<std::uint8_t, sl_bots::ipc::kActionPacketSize> local{};
        const int result = actions_.Read(local.data(), local.size());
        if (result <= 0) {
            return result;
        }
        if (!IsPacketValid(local.data(), local.size(), sl_bots::ipc::kActionPacketSize)) {
            return kProtocolError;
        }
        std::memcpy(packet, local.data(), local.size());
        return 1;
    }

    int SwitchEpoch(std::uint32_t epoch) {
        return observations_.SwitchEpoch(epoch) && actions_.SwitchEpoch(epoch) ? 1 : kOpenError;
    }

    void Abort() {
        observations_.Abort();
        actions_.Abort();
    }

    std::uint64_t Heartbeat() const {
        return std::max(observations_.Heartbeat(), actions_.Heartbeat());
    }

private:
    SharedRing observations_;
    SharedRing actions_;
};

std::mutex g_mutex;
std::unique_ptr<Transport> g_transport;

Transport* CurrentTransport() {
    return g_transport.get();
}

}

extern "C" {

__declspec(dllexport) int SLBots_Open(
    const char* name,
    std::uint32_t epoch,
    std::uint32_t capacity) noexcept {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto transport = std::make_unique<Transport>();
    if (transport->Open(name, epoch, capacity == 0 ? kDefaultCapacity : capacity) < 0) {
        return kOpenError;
    }
    g_transport = std::move(transport);
    return 1;
}

__declspec(dllexport) int SLBots_PublishObservation(const void* packet) noexcept {
    std::lock_guard<std::mutex> lock(g_mutex);
    return CurrentTransport() == nullptr ? kOpenError : CurrentTransport()->PublishObservation(
        packet,
        sl_bots::ipc::kObservationPacketSize);
}

__declspec(dllexport) int SLBots_TryReadAction(void* packet) noexcept {
    std::lock_guard<std::mutex> lock(g_mutex);
    return CurrentTransport() == nullptr ? kOpenError : CurrentTransport()->TryReadAction(
        packet,
        sl_bots::ipc::kActionPacketSize);
}

__declspec(dllexport) std::uint32_t SLBots_GetHeartbeat() noexcept {
    std::lock_guard<std::mutex> lock(g_mutex);
    return CurrentTransport() == nullptr ? 0 : static_cast<std::uint32_t>(CurrentTransport()->Heartbeat());
}

__declspec(dllexport) int SLBots_SwitchEpoch(std::uint32_t epoch) noexcept {
    std::lock_guard<std::mutex> lock(g_mutex);
    return CurrentTransport() == nullptr ? kOpenError : CurrentTransport()->SwitchEpoch(epoch);
}

__declspec(dllexport) void SLBots_Abort() noexcept {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (CurrentTransport() != nullptr) {
        CurrentTransport()->Abort();
    }
}

__declspec(dllexport) void SLBots_Close() noexcept {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (CurrentTransport() != nullptr) {
        CurrentTransport()->Close();
        g_transport.reset();
    }
}

}
