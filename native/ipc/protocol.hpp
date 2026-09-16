#pragma once

#include <cstddef>
#include <cstdint>

namespace sl_bots::ipc {

inline constexpr char kProtocolMagic[8] = {'S', 'L', 'B', 'O', 'T', 'S', '1', '\0'};
inline constexpr std::uint16_t kSchemaVersion = 1;
inline constexpr std::uint16_t kMaxBots = 10;
inline constexpr std::uint16_t kObservationRecordSize = 256;
inline constexpr char kControlMagic[8] = {'S', 'L', 'B', 'C', 'T', 'L', '1', '\0'};
inline constexpr std::uint16_t kControlSchemaVersion = 1;
inline constexpr std::uint16_t kControlEventSize = 64;

#pragma pack(push, 1)
struct ProtocolHeaderV1 {
    char magic[8];
    std::uint16_t schema_version;
    std::uint16_t record_size;
    std::uint32_t epoch;
    std::uint32_t server_tick;
    std::uint16_t bot_count;
    std::uint16_t reserved;
    std::uint64_t write_sequence;
    std::uint32_t crc32;
};

struct BotObservationV1 {
    std::uint8_t payload[kObservationRecordSize];
};

struct BotActionV1 {
    std::int32_t target_tick;
    float forward;
    float side;
    float up;
    float yaw_delta_deg;
    float pitch_delta_deg;
    std::uint32_t buttons;
    std::int16_t weapon_select;
    std::int16_t buy_action;
    std::uint32_t action_valid_mask;
};

struct ControlEventV1 {
    char magic[8];
    std::uint16_t schema_version;
    std::uint16_t event_type;
    std::uint64_t event_sequence;
    std::uint32_t epoch;
    std::int32_t server_tick;
    std::int16_t get5_state;
    std::int16_t map_number;
    std::int16_t round_number;
    std::int16_t team1_score;
    std::int16_t team2_score;
    std::uint32_t policy_generation;
    std::int64_t value_us;
    std::uint64_t match_id_hash;
    std::uint32_t crc32;
    std::uint8_t reserved[2];
};
#pragma pack(pop)

inline constexpr std::uint16_t kHeaderSize = sizeof(ProtocolHeaderV1);
inline constexpr std::uint16_t kActionRecordSize = sizeof(BotActionV1);
inline constexpr std::uint16_t kObservationPacketSize =
    kHeaderSize + kMaxBots * kObservationRecordSize;
inline constexpr std::uint16_t kActionPacketSize =
    kHeaderSize + kMaxBots * kActionRecordSize;

static_assert(sizeof(ProtocolHeaderV1) == 36);
static_assert(offsetof(ProtocolHeaderV1, magic) == 0);
static_assert(offsetof(ProtocolHeaderV1, schema_version) == 8);
static_assert(offsetof(ProtocolHeaderV1, record_size) == 10);
static_assert(offsetof(ProtocolHeaderV1, epoch) == 12);
static_assert(offsetof(ProtocolHeaderV1, server_tick) == 16);
static_assert(offsetof(ProtocolHeaderV1, bot_count) == 20);
static_assert(offsetof(ProtocolHeaderV1, reserved) == 22);
static_assert(offsetof(ProtocolHeaderV1, write_sequence) == 24);
static_assert(offsetof(ProtocolHeaderV1, crc32) == 32);
static_assert(sizeof(BotObservationV1) == kObservationRecordSize);
static_assert(sizeof(BotActionV1) == 36);
static_assert(offsetof(BotActionV1, target_tick) == 0);
static_assert(offsetof(BotActionV1, forward) == 4);
static_assert(offsetof(BotActionV1, side) == 8);
static_assert(offsetof(BotActionV1, up) == 12);
static_assert(offsetof(BotActionV1, yaw_delta_deg) == 16);
static_assert(offsetof(BotActionV1, pitch_delta_deg) == 20);
static_assert(offsetof(BotActionV1, buttons) == 24);
static_assert(offsetof(BotActionV1, weapon_select) == 28);
static_assert(offsetof(BotActionV1, buy_action) == 30);
static_assert(offsetof(BotActionV1, action_valid_mask) == 32);
static_assert(sizeof(ControlEventV1) == kControlEventSize);
static_assert(offsetof(ControlEventV1, magic) == 0);
static_assert(offsetof(ControlEventV1, schema_version) == 8);
static_assert(offsetof(ControlEventV1, event_type) == 10);
static_assert(offsetof(ControlEventV1, event_sequence) == 12);
static_assert(offsetof(ControlEventV1, epoch) == 20);
static_assert(offsetof(ControlEventV1, server_tick) == 24);
static_assert(offsetof(ControlEventV1, policy_generation) == 38);
static_assert(offsetof(ControlEventV1, value_us) == 42);
static_assert(offsetof(ControlEventV1, match_id_hash) == 50);
static_assert(offsetof(ControlEventV1, crc32) == 58);

}  // namespace sl_bots::ipc
