#if defined _sl_bots_ipc_included
    #endinput
#endif
#define _sl_bots_ipc_included

#define SLBOTS_MAX_BOTS 10
#define SLBOTS_OBSERVATION_BYTES 256
#define SLBOTS_OBSERVATION_PACKET_BYTES 2596
#define SLBOTS_OBSERVATION_PACKET_WORDS 649
#define SLBOTS_ACTION_PACKET_BYTES 396
#define SLBOTS_ACTION_PACKET_WORDS 99
#define SLBOTS_HEADER_BYTES 36
#define SLBOTS_ACTION_RECORD_BYTES 36
#define SLBOTS_CONTROL_EVENT_BYTES 64
#define SLBOTS_CONTROL_EVENT_WORDS 16
#define SLBOTS_CONTROL_MAGIC "SLBCTL1"
#define SLBOTS_CONTROL_SCHEMA_VERSION 1
#define SLBOTS_SCHEMA_VERSION 1
#define SLBOTS_MAP_ID_MIRAGE 1

#define SLBOTS_ACTION_MASK_FORWARD (1 << 0)
#define SLBOTS_ACTION_MASK_SIDE (1 << 1)
#define SLBOTS_ACTION_MASK_UP (1 << 2)
#define SLBOTS_ACTION_MASK_YAW (1 << 3)
#define SLBOTS_ACTION_MASK_PITCH (1 << 4)
#define SLBOTS_ACTION_MASK_BUTTONS (1 << 5)
#define SLBOTS_ACTION_MASK_WEAPON (1 << 6)
#define SLBOTS_ACTION_MASK_BUY (1 << 7)

#define SLBOTS_GET5_WARMUP 3
#define SLBOTS_GET5_KNIFE 4
#define SLBOTS_GET5_LIVE 7

#define SLBOTS_CONTROL_RULES_VALIDATED 1
#define SLBOTS_CONTROL_GOING_LIVE 2
#define SLBOTS_CONTROL_LIVE 3
#define SLBOTS_CONTROL_ROUND_START 4
#define SLBOTS_CONTROL_ROUND_END 5
#define SLBOTS_CONTROL_HALFTIME 6
#define SLBOTS_CONTROL_OVERTIME_START 7
#define SLBOTS_CONTROL_PAUSE 8
#define SLBOTS_CONTROL_RESUME 9
#define SLBOTS_CONTROL_BACKUP_RESTORE 10
#define SLBOTS_CONTROL_MAP_END 11
#define SLBOTS_CONTROL_SERIES_END 12
#define SLBOTS_CONTROL_LATENCY_SAMPLE 13
#define SLBOTS_CONTROL_FALLBACK 14

native int SLBots_Open(const char[] name, int epoch, int capacity);
native int SLBots_PublishObservation(int packet[SLBOTS_OBSERVATION_PACKET_WORDS]);
native int SLBots_TryReadAction(int packet[SLBOTS_ACTION_PACKET_WORDS]);
native int SLBots_PublishControl(int packet[SLBOTS_CONTROL_EVENT_WORDS]);
native int SLBots_TryReadControl(int packet[SLBOTS_CONTROL_EVENT_WORDS]);
native int SLBots_GetHeartbeat();
native int SLBots_SwitchEpoch(int epoch);
native void SLBots_Close();

public Extension __ext_sl_bots_ipc =
{
    name = "SL-Bots IPC",
    file = "sl_bots_ipc.ext",
#if defined AUTOLOAD_EXTENSIONS
    autoload = 1,
#else
    autoload = 0,
#endif
#if defined REQUIRE_EXTENSIONS
    required = 1,
#else
    required = 0,
#endif
};
