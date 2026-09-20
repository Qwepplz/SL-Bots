#pragma semicolon 1
#pragma newdecls required

#include <sourcemod>
#include <sdktools>
#include <sdkhooks>
#include <cstrike>
#include <sl_bots_ipc>

native int Get5_GetGameState();
native void Get5_GetMatchID(char[] matchId, int maxlen);
native int Get5_GetMapNumber();

#define SLBOTS_SENSE_CAPACITY 64
#define SLBOTS_MISSING_WINDOW_SIZE 128
#define SLBOTS_LAST_KNOWN_TICKS 16
#define SLBOTS_PLAYER_DIRECT (1 << 0)
#define SLBOTS_PLAYER_RADAR (1 << 1)
#define SLBOTS_PLAYER_AUDIBLE (1 << 2)
#define SLBOTS_PLAYER_ALIVE (1 << 3)
#define SLBOTS_PLAYER_KNOWN (1 << 4)
#define SLBOTS_PLAYER_PERIPHERAL (1 << 5)
#define SLBOTS_MAX_INSTANCE_ID 32
// Actions are applied only within a two-tick publication window.  A packet
// that falls farther behind is stale and must not drive a UserCmd.
#define SLBOTS_MAX_ACTION_AGE_TICKS 2

int g_Epoch = 1;
int g_ControlSequence;
int g_ControlPolicyGeneration;
ConVar g_ControlPolicyGenerationCvar;
int g_LastRulesDiagnosticTick = -1000000;
int g_RoundNumber;
bool g_ControlPaused;
bool g_RulesValid = true;
int g_ObservationSequence;
int g_ObservationSnapshotTick[SLBOTS_SENSE_CAPACITY];
int g_ObservationSnapshotBotCount[SLBOTS_SENSE_CAPACITY];
int g_ObservationSnapshotClients[SLBOTS_SENSE_CAPACITY][SLBOTS_MAX_BOTS];
int g_ObservationSnapshotUserIds[SLBOTS_SENSE_CAPACITY][SLBOTS_MAX_BOTS];
int g_ObservationSnapshotHead;
int g_ObservationSnapshotCount;
int g_BotClients[SLBOTS_MAX_BOTS];
int g_BotCount;
int g_BotStateClients[SLBOTS_MAX_BOTS];
int g_BotStateUserIds[SLBOTS_MAX_BOTS];
int g_LastPublishedTick = -1;
int g_LastActionTick = -1;
int g_LastActionAckTick = -1;
float g_LastActionPacketReceivedTime;
int g_LastActionPacket[SLBOTS_ACTION_PACKET_WORDS];
int g_LastActionClients[SLBOTS_MAX_BOTS];
int g_LastBuyAction[SLBOTS_MAX_BOTS];
bool g_BotPermanentFallback[SLBOTS_MAX_BOTS];
int g_ClientStateUserIds[MAXPLAYERS + 1];
bool g_ClientPermanentFallback[MAXPLAYERS + 1];
int g_MissingConsecutive[SLBOTS_MAX_BOTS];
bool g_MissingWindow[SLBOTS_MAX_BOTS][SLBOTS_MISSING_WINDOW_SIZE];
int g_MissingWindowIndex[SLBOTS_MAX_BOTS];
int g_MissingWindowCount[SLBOTS_MAX_BOTS];
int g_MissingWindowEntries[SLBOTS_MAX_BOTS];
int g_LastMissingTick[SLBOTS_MAX_BOTS];
int g_LastKnownTick[MAXPLAYERS + 1][MAXPLAYERS + 1];
float g_LastKnownPosition[MAXPLAYERS + 1][MAXPLAYERS + 1][3];
int g_SoundCategory[SLBOTS_SENSE_CAPACITY];
int g_SoundSource[SLBOTS_SENSE_CAPACITY];
int g_SoundTick[SLBOTS_SENSE_CAPACITY];
float g_SoundPosition[SLBOTS_SENSE_CAPACITY][3];
int g_SoundHead;
int g_ObservationEventCategory[SLBOTS_SENSE_CAPACITY];
int g_ObservationEventSource[SLBOTS_SENSE_CAPACITY];
int g_ObservationEventTick[SLBOTS_SENSE_CAPACITY];
float g_ObservationEventPosition[SLBOTS_SENSE_CAPACITY][3];
int g_EventHead;
float g_SmokePosition[SLBOTS_SENSE_CAPACITY][3];
int g_SmokeExpiryTick[SLBOTS_SENSE_CAPACITY];
int g_SmokeCount;
char g_IpcName[128];
bool g_Get5Available;
bool g_Get5MatchIdAvailable;
bool g_Get5MapNumberAvailable;
ConVar g_Get5GameStateCvar;
bool g_PlayerFlashedAvailable;
bool g_Open;

public Plugin myinfo = {
    name = "SL-Bots bridge",
    author = "SL-Bots",
    description = "Get5 phase gate and UserCmd bridge for the SL-Bots runtime",
    version = "1.0.0",
    url = ""
};

public void OnPluginStart() {
    RegServerCmd("sl_bots_control_status", Command_ControlStatus);
    CreateConVar("sl_bots_instance_id", "", "Dedicated server instance suffix for SL-Bots IPC", FCVAR_PROTECTED);
    g_ControlPolicyGenerationCvar = CreateConVar(
        "sl_bots_policy_generation",
        "0",
        "Policy generation emitted in SL-Bots telemetry",
        FCVAR_PROTECTED,
        true,
        0.0
    );
    HookConVarChange(g_ControlPolicyGenerationCvar, OnPolicyGenerationChanged);
    RefreshPolicyGeneration();
    MarkNativeAsOptional("Get5_GetMatchID");
    MarkNativeAsOptional("Get5_GetMapNumber");
    HookEvent("round_start", Event_RoundStart, EventHookMode_PostNoCopy);
    HookEvent("round_end", Event_RoundEnd, EventHookMode_Post);
    HookEvent("weapon_fire", Event_Observation, EventHookMode_Post);
    HookEvent("weapon_reload", Event_Observation, EventHookMode_Post);
    HookEvent("player_hurt", Event_Observation, EventHookMode_Post);
    HookEvent("player_death", Event_Observation, EventHookMode_Post);
    g_PlayerFlashedAvailable = HookEventEx("player_flashed", Event_Observation, EventHookMode_Post);
    if (!g_PlayerFlashedAvailable) {
        LogMessage("optional player_flashed event is unavailable; flash observation is degraded");
    }
    HookEvent("grenade_thrown", Event_Observation, EventHookMode_Post);
    HookEvent("smokegrenade_detonate", Event_Observation, EventHookMode_Post);
    HookEvent("flashbang_detonate", Event_Observation, EventHookMode_Post);
    HookEvent("hegrenade_detonate", Event_Observation, EventHookMode_Post);
    HookEvent("inferno_startburn", Event_Observation, EventHookMode_Post);
    HookEvent("decoy_started", Event_Observation, EventHookMode_Post);
    HookEvent("player_footstep", Event_Observation, EventHookMode_Post);
    HookEvent("player_jump", Event_Observation, EventHookMode_Post);
    HookEventEx("get5_going_live", Event_Get5GoingLive, EventHookMode_PostNoCopy);
    HookEventEx("get5_live", Event_Get5Live, EventHookMode_PostNoCopy);
    HookEventEx("get5_pause", Event_Get5Pause, EventHookMode_PostNoCopy);
    HookEventEx("get5_resume", Event_Get5Resume, EventHookMode_PostNoCopy);
    HookEventEx("get5_backup_restore", Event_Get5BackupRestore, EventHookMode_PostNoCopy);
    HookEventEx("get5_map_end", Event_Get5MapEnd, EventHookMode_PostNoCopy);
    HookEventEx("get5_series_end", Event_Get5SeriesEnd, EventHookMode_PostNoCopy);
    g_Get5Available = GetFeatureStatus(FeatureType_Native, "Get5_GetGameState") == FeatureStatus_Available;
    g_Get5MatchIdAvailable = GetFeatureStatus(FeatureType_Native, "Get5_GetMatchID") == FeatureStatus_Available;
    g_Get5MapNumberAvailable = GetFeatureStatus(FeatureType_Native, "Get5_GetMapNumber") == FeatureStatus_Available;
    g_Get5GameStateCvar = FindConVar("get5_game_state");
    g_RulesValid = true;
    g_ControlPaused = false;
    ResetRuntimeState();
    OpenTransport();
}

public Action Command_ControlStatus(int args) {
    PrintToServer("{\"tick\":%d,\"epoch\":%d,\"last_action_tick\":%d}", GetGameTickCount(), g_Epoch, g_LastActionTick);
    for (int slot = 0; slot < SLBOTS_MAX_BOTS; slot++) {
        int client = g_BotClients[slot];
        if (client <= 0 || !IsClientInGame(client) || !IsFakeClient(client)) {
            continue;
        }
        PrintToServer("{\"userid\":%d,\"fallback\":%d,\"missing_consecutive\":%d,\"missing_window\":%d}",
            GetClientUserId(client), g_BotPermanentFallback[slot], g_MissingConsecutive[slot], g_MissingWindowCount[slot]);
    }
    return Plugin_Handled;
}

public void OnMapStart() {
    g_Epoch++;
    g_RoundNumber = 0;
    g_ControlPaused = false;
    g_RulesValid = true;
    g_LastPublishedTick = -1;
    g_LastActionTick = -1;
    RefreshPolicyGeneration();
    ResetRuntimeState();
    OpenTransport();
}

public void OnPolicyGenerationChanged(ConVar convar, const char[] oldValue, const char[] newValue) {
    RefreshPolicyGeneration();
}

void RefreshPolicyGeneration() {
    if (g_ControlPolicyGenerationCvar == null) {
        g_ControlPolicyGeneration = 0;
        return;
    }
    g_ControlPolicyGeneration = GetConVarInt(g_ControlPolicyGenerationCvar);
    if (g_ControlPolicyGeneration < 0) {
        g_ControlPolicyGeneration = 0;
    }
}

public void OnClientPutInServer(int client) {
    ResetKnownClient(client);
}

public void OnClientDisconnect(int client) {
    ResetKnownClient(client);
}

public void OnPluginEnd() {
    if (g_Open) {
        PublishControlEvent(SLBOTS_CONTROL_SERIES_END, GetGameTickCount(), g_RoundNumber);
        SLBots_Close();
        g_Open = false;
    }
}

public void OnMapEnd() {
    if (g_Open) {
        PublishControlEvent(SLBOTS_CONTROL_MAP_END, GetGameTickCount(), g_RoundNumber);
    }
}

void OpenTransport() {
    char mapName[64];
    char instanceId[8];
    GetCurrentMap(mapName, sizeof(mapName));
    instanceId[0] = '\0';
    ConVar instanceCvar = FindConVar("sl_bots_instance_id");
    if (instanceCvar != null) {
        GetConVarString(instanceCvar, instanceId, sizeof(instanceId));
    }
    int numericInstanceId = StringToInt(instanceId);
    if (instanceId[0] != '\0' &&
        (strlen(instanceId) != 2 || numericInstanceId < 1 || numericInstanceId > SLBOTS_MAX_INSTANCE_ID)) {
        LogError("invalid sl_bots_instance_id: %s", instanceId);
        instanceId[0] = '\0';
    }
    if (instanceId[0] == '\0') {
        FormatEx(g_IpcName, sizeof(g_IpcName), "SLBots_%s", mapName);
    } else {
        FormatEx(g_IpcName, sizeof(g_IpcName), "SLBots_%s_%02d", mapName, numericInstanceId);
    }
    if (g_Open) {
        SLBots_Close();
        g_Open = false;
    }
    g_Open = SLBots_Open(g_IpcName, g_Epoch, 64) > 0;
    if (!g_Open) {
        LogError("SL-Bots IPC open failed for %s", g_IpcName);
    }
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast) {
    g_Epoch++;
    g_RoundNumber++;
    g_ControlPaused = false;
    g_LastPublishedTick = -1;
    g_LastActionTick = -1;
    ResetRuntimeState();
    float position[3] = {0.0, 0.0, 0.0};
    RecordObservationEvent(9, 0, position);
    if (g_Open && SLBots_SwitchEpoch(g_Epoch) < 0) {
        OpenTransport();
    }
    PublishControlEvent(SLBOTS_CONTROL_ROUND_START, GetGameTickCount(), g_RoundNumber);
}

public void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast) {
    g_LastPublishedTick = -1;
    g_LastActionTick = -1;
    int winner = GetEventInt(event, "winner");
    int category = 10;
    if (winner == CS_TEAM_T) {
        category = 23;
    } else if (winner == CS_TEAM_CT) {
        category = 24;
    }
    float position[3] = {0.0, 0.0, 0.0};
    RecordObservationEvent(category, 0, position);
    PublishControlEvent(SLBOTS_CONTROL_ROUND_END, GetGameTickCount(), g_RoundNumber);
    int team1Score = GetTeamScore(CS_TEAM_CT);
    int team2Score = GetTeamScore(CS_TEAM_T);
    if (g_RoundNumber == 12) {
        PublishControlEvent(SLBOTS_CONTROL_HALFTIME, GetGameTickCount(), g_RoundNumber);
    } else if (g_RoundNumber == 24 && team1Score == team2Score) {
        PublishControlEvent(SLBOTS_CONTROL_OVERTIME_START, GetGameTickCount(), g_RoundNumber);
    }
}

public void Event_Get5GoingLive(Event event, const char[] name, bool dontBroadcast) {
    if (!ValidateMR12Rules()) {
        g_RulesValid = false;
        PublishControlEvent(SLBOTS_CONTROL_FALLBACK, GetGameTickCount(), g_RoundNumber);
        return;
    }
    g_RulesValid = true;
    PublishControlEvent(SLBOTS_CONTROL_RULES_VALIDATED, GetGameTickCount(), g_RoundNumber);
    PublishControlEvent(SLBOTS_CONTROL_GOING_LIVE, GetGameTickCount(), g_RoundNumber);
}

public void Event_Get5Live(Event event, const char[] name, bool dontBroadcast) {
    g_RulesValid = ValidateMR12Rules();
    if (g_RulesValid) {
        PublishControlEvent(SLBOTS_CONTROL_LIVE, GetGameTickCount(), g_RoundNumber);
    } else {
        PublishControlEvent(SLBOTS_CONTROL_FALLBACK, GetGameTickCount(), g_RoundNumber);
    }
}

public void Event_Get5Pause(Event event, const char[] name, bool dontBroadcast) {
    g_ControlPaused = true;
    PublishControlEvent(SLBOTS_CONTROL_PAUSE, GetGameTickCount(), g_RoundNumber);
}

public void Event_Get5Resume(Event event, const char[] name, bool dontBroadcast) {
    g_ControlPaused = false;
    PublishControlEvent(SLBOTS_CONTROL_RESUME, GetGameTickCount(), g_RoundNumber);
}

public void Event_Get5BackupRestore(Event event, const char[] name, bool dontBroadcast) {
    g_Epoch++;
    ResetRuntimeState();
    if (g_Open && SLBots_SwitchEpoch(g_Epoch) < 0) {
        OpenTransport();
    }
    PublishControlEvent(SLBOTS_CONTROL_BACKUP_RESTORE, GetGameTickCount(), g_RoundNumber);
}

public void Event_Get5MapEnd(Event event, const char[] name, bool dontBroadcast) {
    g_RulesValid = false;
    PublishControlEvent(SLBOTS_CONTROL_MAP_END, GetGameTickCount(), g_RoundNumber);
}

public void Event_Get5SeriesEnd(Event event, const char[] name, bool dontBroadcast) {
    g_RulesValid = false;
    PublishControlEvent(SLBOTS_CONTROL_SERIES_END, GetGameTickCount(), g_RoundNumber);
}

bool ValidateMR12Rules() {
    return ValidateCvar("mp_maxrounds", 24) &&
        ValidateCvar("mp_halftime", 1) &&
        ValidateCvar("mp_match_can_clinch", 1) &&
        ValidateCvar("mp_overtime_enable", 1) &&
        ValidateCvar("mp_overtime_maxrounds", 6) &&
        ValidateCvar("mp_overtime_startmoney", 10000);
}

bool ValidateCvar(const char[] name, int expected) {
    ConVar cvar = FindConVar(name);
    if (cvar == null) {
        return false;
    }
    int actual = GetConVarInt(cvar);
    if (actual != expected) {
        int currentTick = GetGameTickCount();
        if (currentTick - g_LastRulesDiagnosticTick >= 128) {
            LogError("MR12 cvar mismatch: %s expected %d actual %d", name, expected, actual);
            g_LastRulesDiagnosticTick = currentTick;
        }
        return false;
    }
    return true;
}

void ResetRuntimeState() {
    g_BotCount = 0;
    g_LastActionAckTick = -1;
    g_LastActionPacketReceivedTime = 0.0;
    g_ObservationSnapshotHead = 0;
    g_ObservationSnapshotCount = 0;
    g_SoundHead = 0;
    g_EventHead = 0;
    g_SmokeCount = 0;
    for (int slot = 0; slot < SLBOTS_MAX_BOTS; slot++) {
        g_BotClients[slot] = 0;
        g_BotStateClients[slot] = 0;
        g_BotStateUserIds[slot] = 0;
        g_LastActionClients[slot] = 0;
        g_LastBuyAction[slot] = 0;
        g_BotPermanentFallback[slot] = false;
        g_MissingConsecutive[slot] = 0;
        g_MissingWindowIndex[slot] = 0;
        g_MissingWindowCount[slot] = 0;
        g_MissingWindowEntries[slot] = 0;
        g_LastMissingTick[slot] = -1;
        for (int index = 0; index < SLBOTS_MISSING_WINDOW_SIZE; index++) {
            g_MissingWindow[slot][index] = false;
        }
    }
    for (int index = 0; index < SLBOTS_ACTION_PACKET_WORDS; index++) {
        g_LastActionPacket[index] = 0;
    }
    for (int index = 0; index < SLBOTS_SENSE_CAPACITY; index++) {
        g_ObservationSnapshotTick[index] = -1;
        g_ObservationSnapshotBotCount[index] = 0;
        for (int slot = 0; slot < SLBOTS_MAX_BOTS; slot++) {
            g_ObservationSnapshotClients[index][slot] = 0;
            g_ObservationSnapshotUserIds[index][slot] = 0;
        }
    }
    for (int observer = 1; observer <= MaxClients; observer++) {
        g_ClientStateUserIds[observer] = 0;
        g_ClientPermanentFallback[observer] = false;
        for (int target = 1; target <= MaxClients; target++) {
            g_LastKnownTick[observer][target] = -1;
            for (int axis = 0; axis < 3; axis++) {
                g_LastKnownPosition[observer][target][axis] = 0.0;
            }
        }
    }
}

void ResetKnownClient(int client) {
    if (client < 1 || client > MaxClients) {
        return;
    }
    g_ClientStateUserIds[client] = 0;
    g_ClientPermanentFallback[client] = false;
    for (int peer = 1; peer <= MaxClients; peer++) {
        g_LastKnownTick[client][peer] = -1;
        g_LastKnownTick[peer][client] = -1;
        for (int axis = 0; axis < 3; axis++) {
            g_LastKnownPosition[client][peer][axis] = 0.0;
            g_LastKnownPosition[peer][client][axis] = 0.0;
        }
    }
    for (int slot = 0; slot < SLBOTS_MAX_BOTS; slot++) {
        if (g_LastActionClients[slot] == client) {
            g_LastActionClients[slot] = 0;
        }
    }
}

public void OnGameFrame() {
    if (!g_Open) {
        return;
    }
    int phase = GetActivePhase();
    if (phase < 0) {
        return;
    }
    int serverTick = GetGameTickCount();
    if (serverTick == g_LastPublishedTick) {
        return;
    }
    int packet[SLBOTS_OBSERVATION_PACKET_WORDS];
    BuildObservationPacket(packet, serverTick, phase);
    if (SLBots_PublishObservation(packet) > 0) {
        g_LastPublishedTick = serverTick;
        RecordObservationSnapshot(serverTick);
        g_ObservationSequence++;
    }
    PollActionPacket();
}

public Action OnPlayerRunCmd(
    int client,
    int &buttons,
    int &impulse,
    float vel[3],
    float angles[3],
    int &weapon,
    int &subtype,
    int &cmdnum,
    int &tickcount,
    int &seed,
    int mouse[2]) {
    if (!g_Open || !IsClientInGame(client) || !IsFakeClient(client)) {
        return Plugin_Continue;
    }
    if (GetActivePhase() < 0) {
        return Plugin_Continue;
    }
    int slot = FindBotSlot(client);
    if (slot < 0) {
        return Plugin_Continue;
    }
    if (g_BotPermanentFallback[slot]) {
        return Plugin_Continue;
    }
    // Fake clients report tickcount=0 in OnPlayerRunCmd.  The missing-action
    // window must advance on the server tick or a silent worker can be
    // deduplicated forever at tick 0.
    int commandTick = GetGameTickCount();
    int actionSlot = FindActionSlot(client);
    if (actionSlot < 0) {
        RecordMissing(slot, commandTick);
        if (g_BotPermanentFallback[slot]) {
            return Plugin_Continue;
        }
        ApplyNeutralAction(client, buttons, impulse, vel, angles, weapon, subtype, mouse);
        return Plugin_Changed;
    }
    int actionOffset = 9 + actionSlot * 9;
    int targetTick = g_LastActionPacket[actionOffset];
    // In the real server OnPlayerRunCmd can run before the Python worker's
    // action for this observation is published.  Keep the latest action for
    // a short bounded window instead of converting every one-tick late action
    // into a neutral command.  A genuinely stale worker still falls back.
    if (g_LastActionTick < 0 || targetTick > commandTick ||
        commandTick - targetTick > SLBOTS_MAX_ACTION_AGE_TICKS) {
        RecordMissing(slot, commandTick);
        if (g_BotPermanentFallback[slot]) {
            return Plugin_Continue;
        }
        ApplyNeutralAction(client, buttons, impulse, vel, angles, weapon, subtype, mouse);
        return Plugin_Changed;
    }
    int validMask = g_LastActionPacket[actionOffset + 8];
    if (validMask == 0) {
        SetPermanentFallback(slot);
        return Plugin_Continue;
    }
    RecordValid(slot, commandTick);
    float currentAngles[3];
    GetClientEyeAngles(client, currentAngles);
    angles[0] = currentAngles[0];
    angles[1] = currentAngles[1];
    angles[2] = currentAngles[2];
    mouse[0] = 0;
    mouse[1] = 0;
    impulse = 0;
    subtype = 0;
    int currentWeapon = CurrentWeaponEntity(client);
    if (currentWeapon > 0) {
        weapon = currentWeapon;
    }
    if (validMask & SLBOTS_ACTION_MASK_FORWARD) {
        vel[0] = GetActionFloat(actionOffset + 1) * 450.0;
    } else {
        vel[0] = 0.0;
    }
    if (validMask & SLBOTS_ACTION_MASK_SIDE) {
        vel[1] = GetActionFloat(actionOffset + 2) * 450.0;
    } else {
        vel[1] = 0.0;
    }
    if (validMask & SLBOTS_ACTION_MASK_UP) {
        vel[2] = GetActionFloat(actionOffset + 3) * 320.0;
    } else {
        vel[2] = 0.0;
    }
    if (validMask & SLBOTS_ACTION_MASK_YAW) {
        angles[1] = NormalizeYaw(currentAngles[1] + GetActionFloat(actionOffset + 4));
    }
    if (validMask & SLBOTS_ACTION_MASK_PITCH) {
        angles[0] = NormalizePitch(currentAngles[0] + GetActionFloat(actionOffset + 5));
    }
    if (validMask & SLBOTS_ACTION_MASK_BUTTONS) {
        buttons = g_LastActionPacket[actionOffset + 6];
    } else {
        buttons = 0;
    }
    if (validMask & SLBOTS_ACTION_MASK_WEAPON) {
        int selectedWeapon = Signed16(g_LastActionPacket[actionOffset + 7]);
        if (selectedWeapon >= 0) {
            int selectedEntity = WeaponEntityForModelId(client, selectedWeapon);
            if (selectedEntity > 0) {
                weapon = selectedEntity;
            }
        }
    }
    if (validMask & SLBOTS_ACTION_MASK_BUY) {
        ExecuteBuyAction(client, slot, Signed16(g_LastActionPacket[actionOffset + 7] >> 16));
    }
    RecordActionApplied(commandTick);
    return Plugin_Changed;
}

int GetActivePhase() {
    if (g_ControlPaused) {
        return -1;
    }
    if (g_Get5GameStateCvar == null) {
        g_Get5GameStateCvar = FindConVar("get5_game_state");
    }
    if (!g_Get5Available && g_Get5GameStateCvar == null) {
        return -1;
    }
    int state = g_Get5Available ? Get5_GetGameState() : -1;
    if (g_Get5GameStateCvar != null) {
        state = GetConVarInt(g_Get5GameStateCvar);
    }
    if (state == SLBOTS_GET5_LIVE) {
        // Get5 can emit going_live while its asynchronous live.cfg replay is
        // still settling. Revalidate on the live state so a transient warmup
        // cvar mismatch does not permanently suppress live observations.
        g_RulesValid = ValidateMR12Rules();
        return g_RulesValid ? 2 : -1;
    }
    if (!g_RulesValid) {
        return -1;
    }
    if (state == SLBOTS_GET5_WARMUP) {
        return 0;
    }
    if (state == SLBOTS_GET5_KNIFE) {
        return 1;
    }
    return -1;
}

void BuildObservationPacket(int packet[SLBOTS_OBSERVATION_PACKET_WORDS], int serverTick, int phase) {
    for (int index = 0; index < SLBOTS_OBSERVATION_PACKET_WORDS; index++) {
        packet[index] = 0;
    }
    SetPacketInt32(packet, 0, 0x4F424C53);
    SetPacketInt32(packet, 4, 0x00315354);
    SetPacketInt16(packet, 8, SLBOTS_SCHEMA_VERSION);
    SetPacketInt16(packet, 10, SLBOTS_OBSERVATION_PACKET_BYTES);
    SetPacketInt32(packet, 12, g_Epoch);
    SetPacketInt32(packet, 16, serverTick);
    g_BotCount = CollectBots();
    SetPacketInt16(packet, 20, g_BotCount);
    SetPacketInt16(packet, 22, 0);
    SetPacketInt32(packet, 24, g_ObservationSequence);
    SetPacketInt32(packet, 28, 0);
    for (int slot = 0; slot < g_BotCount; slot++) {
        int client = g_BotClients[slot];
        int base = SLBOTS_HEADER_BYTES + slot * SLBOTS_OBSERVATION_BYTES;
        BuildObservation(packet, base, client, phase);
    }
    SetPacketInt32(packet, 32, 0);
    SetPacketInt32(packet, 32, Crc32(packet, SLBOTS_OBSERVATION_PACKET_BYTES));
}

void BuildObservation(int packet[SLBOTS_OBSERVATION_PACKET_WORDS], int base, int client, int phase) {
    SetPacketByte(packet, base + 0, 'O');
    SetPacketByte(packet, base + 1, 'B');
    SetPacketByte(packet, base + 2, 'S');
    SetPacketByte(packet, base + 3, '1');
    SetPacketByte(packet, base + 4, 1);
    SetPacketByte(packet, base + 5, phase);
    SetPacketByte(packet, base + 6, GetClientTeam(client));
    SetPacketByte(packet, base + 7, SLBOTS_MAP_ID_MIRAGE);
    float origin[3];
    float eyeAngles[3];
    float velocity[3];
    GetClientAbsOrigin(client, origin);
    GetClientEyeAngles(client, eyeAngles);
    GetEntPropVector(client, Prop_Data, "m_vecVelocity", velocity);
    int connectionToken = GetClientUserId(client) & 0xFFFF;
    SetPacketInt16(packet, base + 8, connectionToken > 0 ? connectionToken : client);
    SetPacketByte(packet, base + 10, GetClientHealth(client));
    SetPacketByte(packet, base + 11, GetEntProp(client, Prop_Send, "m_ArmorValue"));
    SetPacketInt16(packet, base + 12, GetEntProp(client, Prop_Send, "m_iAccount"));
    SetPacketInt16(packet, base + 14, RoundToNearest(origin[0]));
    SetPacketInt16(packet, base + 16, RoundToNearest(origin[1]));
    SetPacketInt16(packet, base + 18, RoundToNearest(origin[2]));
    SetPacketInt16(packet, base + 20, RoundToNearest(eyeAngles[1] * 10.0));
    SetPacketInt16(packet, base + 22, RoundToNearest(eyeAngles[0] * 10.0));
    SetPacketByte(packet, base + 24, ClampInt(RoundToNearest(velocity[0] / 16.0), -128, 127));
    SetPacketByte(packet, base + 25, ClampInt(RoundToNearest(velocity[1] / 16.0), -128, 127));
    SetPacketByte(packet, base + 26, EncodeSelfFlags(client));
    SetPacketByte(packet, base + 27, EncodeSelfEffects(client));
    SetPacketByte(packet, base + 28, 191);
    SetPacketByte(packet, base + 29, 191);
    SetPacketByte(packet, base + 30, 184);
    SetPacketByte(packet, base + 31, 166);
    SetPacketByte(packet, base + 32, 96 + (client * 17) % 96);
    SetPacketByte(packet, base + 33, 128 + (client * 29) % 96);
    SetPacketByte(packet, base + 34, 96 + (client * 43) % 96);
    SetPacketByte(packet, base + 35, 128 + (client * 11) % 96);
    SetPacketByte(packet, base + 36, 0);
    SetPacketByte(packet, base + 37, 0);
    SetPacketByte(packet, base + 38, EncodeFlashRecovery(client));
    SetPacketByte(packet, base + 39, EncodeSmokeOcclusion(client));
    SetPacketByte(packet, base + 40, 0);
    SetPacketByte(packet, base + 41, GetActiveWeaponId(client));
    SetPacketByte(packet, base + 42, phase == 2 ? 1 : 0);
    SetPacketByte(packet, base + 43, 0);
    int currentTick = GetGameTickCount();
    int playerSlot = 0;
    for (int target = 1; target <= MaxClients && playerSlot < 9; target++) {
        if (target == client || !IsClientInGame(target) || !IsPlayerAlive(target)) {
            continue;
        }
        int targetTeam = GetClientTeam(target);
        bool teammate = targetTeam == GetClientTeam(client);
        bool direct = CanSeeClient(client, target);
        bool radar = IsRadarSpotted(target);
        bool audible = HasRecentSound(target, client);
        if (!teammate && !direct && !radar && !audible) {
            int lastKnownTick = g_LastKnownTick[client][target];
            if (lastKnownTick < 0 || currentTick - lastKnownTick > SLBOTS_LAST_KNOWN_TICKS) {
                continue;
            }
        }
        if (direct) {
            float targetPosition[3];
            GetClientAbsOrigin(target, targetPosition);
            g_LastKnownTick[client][target] = currentTick;
            for (int axis = 0; axis < 3; axis++) {
                g_LastKnownPosition[client][target][axis] = targetPosition[axis];
            }
        }
        WritePlayerObservation(
            packet,
            base + 44 + playerSlot * 12,
            client,
            target,
            teammate,
            direct,
            radar,
            audible,
            !teammate && !direct && !radar && !audible,
            !teammate && !direct && !radar && audible
        );
        playerSlot++;
    }
    BuildSoundSlots(packet, base, client);
    BuildObservationEvents(packet, base, client);
    BuildCollisionRays(packet, base, client);
}

void WritePlayerObservation(
    int packet[SLBOTS_OBSERVATION_PACKET_WORDS],
    int offset,
    int observer,
    int target,
    bool teammate,
    bool direct,
    bool radar,
    bool audible,
    bool delayed,
    bool soundOnly
) {
    float observerOrigin[3];
    float targetOrigin[3];
    float observerAngles[3];
    float direction[3];
    float targetAngles[3];
    GetClientEyePosition(observer, observerOrigin);
    if (soundOnly) {
        float soundAge;
        if (!FindRecentSound(target, observer, targetOrigin, soundAge)) {
            targetOrigin[0] = observerOrigin[0];
            targetOrigin[1] = observerOrigin[1];
            targetOrigin[2] = observerOrigin[2];
        }
    } else if (delayed && !direct && !radar && !audible) {
        for (int axis = 0; axis < 3; axis++) {
            targetOrigin[axis] = g_LastKnownPosition[observer][target][axis];
        }
    } else {
        GetClientAbsOrigin(target, targetOrigin);
    }
    GetClientEyeAngles(observer, observerAngles);
    MakeVectorFromPoints(observerOrigin, targetOrigin, direction);
    GetVectorAngles(direction, targetAngles);
    float bearing = NormalizeYaw(targetAngles[1] - observerAngles[1]);
    float distance = GetVectorDistance(observerOrigin, targetOrigin);
    int flags = 0;
    if (direct) {
        flags |= SLBOTS_PLAYER_DIRECT;
    }
    if (radar) {
        flags |= SLBOTS_PLAYER_RADAR;
    }
    if (audible) {
        flags |= SLBOTS_PLAYER_AUDIBLE;
    }
    if (IsPlayerAlive(target)) {
        flags |= SLBOTS_PLAYER_ALIVE;
    }
    if (teammate || direct || delayed) {
        flags |= SLBOTS_PLAYER_KNOWN;
    }
    if (!direct) {
        flags |= SLBOTS_PLAYER_PERIPHERAL;
    }
    SetPacketInt16(packet, offset + 0, soundOnly ? 0 : target);
    SetPacketByte(packet, offset + 2, teammate ? 1 : -1);
    SetPacketByte(packet, offset + 3, flags);
    SetPacketInt16(packet, offset + 4, ClampInt(RoundToNearest(bearing * 100.0), -32768, 32767));
    SetPacketByte(packet, offset + 6, ClampInt(RoundToNearest(NormalizeYaw(targetAngles[0] - observerAngles[0])), -128, 127));
    SetPacketInt16(packet, offset + 7, ClampInt(RoundToNearest(distance), 0, 65535));
    int ageDeciseconds = delayed ? ClampInt(
        RoundToNearest(float(GetGameTickCount() - g_LastKnownTick[observer][target]) * 10.0 / 128.0),
        0,
        255
    ) : 0;
    SetPacketByte(
        packet,
        offset + 9,
        audible ? ClampInt(RoundToNearest(GetRecentSoundAge(target, observer) * 10.0), 0, 255) : ageDeciseconds
    );
    SetPacketByte(packet, offset + 10, teammate || direct ? 255 : 160);
    SetPacketByte(packet, offset + 11, direct ? 255 : (audible ? 128 : 0));
}

bool CanSeeClient(int observer, int target) {
    float start[3];
    float end[3];
    if (IsClientFlashed(observer) || !IsWithinViewCone(observer, target)) {
        return false;
    }
    GetClientEyePosition(observer, start);
    GetClientEyePosition(target, end);
    if (SmokeLineOccluded(start, end)) {
        return false;
    }
    Handle trace = TR_TraceRayFilterEx(start, end, MASK_VISIBLE, RayType_EndPoint, TraceFilterIgnorePlayers);
    bool visible = !TR_DidHit(trace);
    delete trace;
    return visible;
}

public bool TraceFilterIgnorePlayers(int entity, int contentsMask, any data) {
    return entity <= 0 || entity > MaxClients;
}

bool IsClientFlashed(int client) {
    if (HasEntProp(client, Prop_Send, "m_flFlashBangTime") &&
        HasEntProp(client, Prop_Send, "m_flFlashDuration")) {
        float remaining = GetEntPropFloat(client, Prop_Send, "m_flFlashBangTime") - GetGameTime();
        if (remaining > 0.0) {
            return true;
        }
    }
    return HasEntProp(client, Prop_Send, "m_flFlashMaxAlpha") &&
        GetEntPropFloat(client, Prop_Send, "m_flFlashMaxAlpha") > 16.0;
}

bool IsWithinViewCone(int observer, int target) {
    float observerPosition[3];
    float targetPosition[3];
    float angles[3];
    float viewForward[3];
    float right[3];
    float up[3];
    float direction[3];
    GetClientEyePosition(observer, observerPosition);
    GetClientEyePosition(target, targetPosition);
    GetClientEyeAngles(observer, angles);
    GetAngleVectors(angles, viewForward, right, up);
    MakeVectorFromPoints(observerPosition, targetPosition, direction);
    if (NormalizeVector(direction, direction) <= 0.0) {
        return true;
    }
    return GetVectorDotProduct(viewForward, direction) >= 0.15;
}

void PruneSmokeStates() {
    int currentTick = GetGameTickCount();
    int writeIndex = 0;
    for (int index = 0; index < g_SmokeCount; index++) {
        if (g_SmokeExpiryTick[index] <= currentTick) {
            continue;
        }
        if (writeIndex != index) {
            g_SmokeExpiryTick[writeIndex] = g_SmokeExpiryTick[index];
            for (int axis = 0; axis < 3; axis++) {
                g_SmokePosition[writeIndex][axis] = g_SmokePosition[index][axis];
            }
        }
        writeIndex++;
    }
    g_SmokeCount = writeIndex;
}

void RecordSmoke(const float position[3]) {
    PruneSmokeStates();
    int index = g_SmokeCount;
    if (index >= SLBOTS_SENSE_CAPACITY) {
        index = SLBOTS_SENSE_CAPACITY - 1;
        for (int move = 1; move < SLBOTS_SENSE_CAPACITY; move++) {
            g_SmokeExpiryTick[move - 1] = g_SmokeExpiryTick[move];
            for (int axis = 0; axis < 3; axis++) {
                g_SmokePosition[move - 1][axis] = g_SmokePosition[move][axis];
            }
        }
    } else {
        g_SmokeCount++;
    }
    for (int axis = 0; axis < 3; axis++) {
        g_SmokePosition[index][axis] = position[axis];
    }
    g_SmokeExpiryTick[index] = GetGameTickCount() + RoundToCeil(18.0 * 128.0);
}

float SmokeOcclusionAt(const float point[3]) {
    PruneSmokeStates();
    float result = 0.0;
    for (int index = 0; index < g_SmokeCount; index++) {
        float distance = GetVectorDistance(point, g_SmokePosition[index]);
        if (distance >= 144.0) {
            continue;
        }
        float candidate = ClampFloat(1.0 - distance / 144.0, 0.0, 1.0);
        if (candidate > result) {
            result = candidate;
        }
    }
    return result;
}

bool SmokeLineOccluded(const float start[3], const float end[3]) {
    PruneSmokeStates();
    float line[3];
    MakeVectorFromPoints(start, end, line);
    float denominator = GetVectorDotProduct(line, line);
    if (denominator <= 0.0) {
        return SmokeOcclusionAt(start) > 0.0;
    }
    for (int index = 0; index < g_SmokeCount; index++) {
        float toSmoke[3];
        MakeVectorFromPoints(start, g_SmokePosition[index], toSmoke);
        float fraction = ClampFloat(GetVectorDotProduct(toSmoke, line) / denominator, 0.0, 1.0);
        float closest[3];
        for (int axis = 0; axis < 3; axis++) {
            closest[axis] = start[axis] + line[axis] * fraction;
        }
        if (GetVectorDistance(closest, g_SmokePosition[index]) < 144.0) {
            return true;
        }
    }
    return false;
}

int EncodeFlashRecovery(int client) {
    if (HasEntProp(client, Prop_Send, "m_flFlashBangTime") &&
        HasEntProp(client, Prop_Send, "m_flFlashDuration")) {
        float duration = GetEntPropFloat(client, Prop_Send, "m_flFlashDuration");
        float remaining = GetEntPropFloat(client, Prop_Send, "m_flFlashBangTime") - GetGameTime();
        if (duration > 0.0 && remaining > 0.0) {
            return ClampInt(RoundToNearest((1.0 - remaining / duration) * 255.0), 0, 255);
        }
        return 255;
    }
    if (!HasEntProp(client, Prop_Send, "m_flFlashMaxAlpha")) {
        return 255;
    }
    return GetEntPropFloat(client, Prop_Send, "m_flFlashMaxAlpha") <= 16.0 ? 255 : 0;
}

int EncodeSmokeOcclusion(int client) {
    float position[3];
    GetClientEyePosition(client, position);
    return ClampInt(RoundToNearest(SmokeOcclusionAt(position) * 255.0), 0, 255);
}

int EncodeSelfFlags(int client) {
    // ObservationProjectionV1 bits are not Source engine entity flags.
    int flags = 0;
    int entityFlags = GetEntityFlags(client);
    if (IsPlayerAlive(client)) { flags |= 1 << 0; }
    if ((entityFlags & FL_DUCKING) != 0) { flags |= 1 << 1; }
    if (HasEntProp(client, Prop_Send, "m_bIsWalking") && GetEntProp(client, Prop_Send, "m_bIsWalking")) { flags |= 1 << 2; }
    if (HasEntProp(client, Prop_Send, "m_bIsScoped") && GetEntProp(client, Prop_Send, "m_bIsScoped")) { flags |= 1 << 3; }
    if (IsPlayerAlive(client) && (entityFlags & FL_ONGROUND) == 0) { flags |= 1 << 4; }
    if (EncodeFlashRecovery(client) < 255) { flags |= 1 << 5; }
    if (HasEntProp(client, Prop_Send, "m_bIsDefusing") && GetEntProp(client, Prop_Send, "m_bIsDefusing")) { flags |= 1 << 6; }
    int weapon = CurrentWeaponEntity(client);
    if (weapon > MaxClients && IsValidEntity(weapon) && HasEntProp(weapon, Prop_Send, "m_bStartedArming") && GetEntProp(weapon, Prop_Send, "m_bStartedArming")) { flags |= 1 << 7; }
    return flags;
}

int EncodeSelfEffects(int client) {
    int flash = ClampInt(RoundToNearest(float(EncodeFlashRecovery(client)) / 255.0 * 63.0), 0, 63);
    int smoke = ClampInt(RoundToNearest(float(EncodeSmokeOcclusion(client)) / 255.0 * 3.0), 0, 3);
    return flash | (smoke << 6);
}

int GetActiveWeaponId(int client) {
    return WeaponModelId(CurrentWeaponEntity(client));
}

int CurrentWeaponEntity(int client) {
    if (!HasEntProp(client, Prop_Send, "m_hActiveWeapon")) {
        return -1;
    }
    int weapon = GetEntPropEnt(client, Prop_Send, "m_hActiveWeapon");
    return weapon > MaxClients && IsValidEntity(weapon) ? weapon : -1;
}

int WeaponModelId(int weapon) {
    if (weapon <= MaxClients || !IsValidEntity(weapon) || !HasEntProp(weapon, Prop_Send, "m_iItemDefinitionIndex")) {
        return 0;
    }
    char classname[64];
    GetEntityClassname(weapon, classname, sizeof(classname));
    if (StrEqual(classname, "weapon_c4")) {
        return 14;
    }
    int definition = GetEntProp(weapon, Prop_Send, "m_iItemDefinitionIndex");
    switch (definition) {
        case 4: return 1;
        case 61: return 2;
        case 32: return 3;
        case 7: return 4;
        case 16: return 5;
        case 60: return 6;
        case 9: return 7;
        case 40: return 8;
        case 1: return 9;
        case 42, 59: return 10;
        case 44: return 11;
        case 43: return 12;
        case 45: return 13;
    }
    return 0;
}

int WeaponEntityForModelId(int client, int modelId) {
    int currentWeapon = CurrentWeaponEntity(client);
    if (modelId <= 0) {
        return currentWeapon;
    }
    for (int weaponSlot = 0; weaponSlot <= 5; weaponSlot++) {
        int weapon = GetPlayerWeaponSlot(client, weaponSlot);
        if (weapon > MaxClients && IsValidEntity(weapon) && WeaponModelId(weapon) == modelId) {
            return weapon;
        }
    }
    return currentWeapon;
}

bool IsRadarSpotted(int target) {
    return HasEntProp(target, Prop_Send, "m_bSpotted") && GetEntProp(target, Prop_Send, "m_bSpotted") != 0;
}

bool FindRecentSound(int source, int observer, float position[3], float &ageSeconds) {
    int currentTick = GetGameTickCount();
    float observerPosition[3];
    GetClientEyePosition(observer, observerPosition);
    int count = g_SoundHead < SLBOTS_SENSE_CAPACITY ? g_SoundHead : SLBOTS_SENSE_CAPACITY;
    ageSeconds = 4.0;
    for (int axis = 0; axis < 3; axis++) {
        position[axis] = 0.0;
    }
    for (int age = 0; age < count; age++) {
        int index = (g_SoundHead - 1 - age) % SLBOTS_SENSE_CAPACITY;
        if (index < 0) {
            index += SLBOTS_SENSE_CAPACITY;
        }
        if (g_SoundSource[index] != source || currentTick - g_SoundTick[index] > 4 * 128) {
            continue;
        }
        if (GetVectorDistance(observerPosition, g_SoundPosition[index]) <= 1800.0) {
            for (int axis = 0; axis < 3; axis++) {
                position[axis] = g_SoundPosition[index][axis];
            }
            ageSeconds = float(currentTick - g_SoundTick[index]) / 128.0;
            return true;
        }
    }
    return false;
}

bool HasRecentSound(int source, int observer) {
    float position[3];
    float ageSeconds;
    return FindRecentSound(source, observer, position, ageSeconds);
}

float GetRecentSoundAge(int source, int observer) {
    float position[3];
    float ageSeconds;
    if (FindRecentSound(source, observer, position, ageSeconds)) {
        return ageSeconds;
    }
    return 4.0;
}

int ResolveEventSource(Event event, const char[] name) {
    int userId = GetEventInt(event, "userid");
    if (StrEqual(name, "player_hurt") || StrEqual(name, "player_death") || StrEqual(name, "player_flashed")) {
        int attacker = GetEventInt(event, "attacker");
        if (attacker > 0) {
            userId = attacker;
        }
    }
    return GetClientOfUserId(userId);
}

void ResolveEventPosition(Event event, int source, float position[3]) {
    if (source > 0 && IsClientInGame(source)) {
        GetClientAbsOrigin(source, position);
        return;
    }
    position[0] = GetEventFloat(event, "x");
    position[1] = GetEventFloat(event, "y");
    position[2] = GetEventFloat(event, "z");
}

int SoundIdForEvent(const char[] name) {
    if (StrEqual(name, "player_footstep")) {
        return 1;
    }
    if (StrEqual(name, "weapon_fire")) {
        return 2;
    }
    if (StrEqual(name, "weapon_reload")) {
        return 6;
    }
    if (StrEqual(name, "player_jump")) {
        return 7;
    }
    if (StrEqual(name, "smokegrenade_detonate")) {
        return 5;
    }
    if (StrEqual(name, "flashbang_detonate")) {
        return 4;
    }
    if (StrContains(name, "grenade", false) >= 0 || StrEqual(name, "inferno_startburn") || StrEqual(name, "decoy_started")) {
        return 3;
    }
    return 0;
}

int ObservationEventId(const char[] name) {
    if (StrEqual(name, "weapon_fire")) {
        return 1;
    }
    if (StrEqual(name, "weapon_reload")) {
        return 2;
    }
    if (StrEqual(name, "player_hurt")) {
        return 3;
    }
    if (StrEqual(name, "player_flashed")) {
        return 4;
    }
    if (StrEqual(name, "player_death")) {
        return 5;
    }
    if (StrEqual(name, "grenade_thrown")) {
        return 6;
    }
    if (StrEqual(name, "smokegrenade_detonate")) {
        return 13;
    }
    if (StrEqual(name, "flashbang_detonate")) {
        return 15;
    }
    if (StrEqual(name, "inferno_startburn")) {
        return 16;
    }
    if (StrEqual(name, "hegrenade_detonate")) {
        return 18;
    }
    if (StrEqual(name, "decoy_started")) {
        return 19;
    }
    if (StrEqual(name, "player_footstep")) {
        return 21;
    }
    if (StrEqual(name, "player_jump")) {
        return 22;
    }
    return 0;
}

void RecordSound(int category, int source, const float position[3]) {
    int index = g_SoundHead % SLBOTS_SENSE_CAPACITY;
    g_SoundCategory[index] = category;
    g_SoundSource[index] = source;
    g_SoundTick[index] = GetGameTickCount();
    for (int axis = 0; axis < 3; axis++) {
        g_SoundPosition[index][axis] = position[axis];
    }
    g_SoundHead++;
}

void RecordObservationEvent(int category, int source, const float position[3]) {
    int index = g_EventHead % SLBOTS_SENSE_CAPACITY;
    g_ObservationEventCategory[index] = category;
    g_ObservationEventSource[index] = source;
    g_ObservationEventTick[index] = GetGameTickCount();
    for (int axis = 0; axis < 3; axis++) {
        g_ObservationEventPosition[index][axis] = position[axis];
    }
    g_EventHead++;
}

public void Event_Observation(Event event, const char[] name, bool dontBroadcast) {
    int source = ResolveEventSource(event, name);
    float position[3];
    ResolveEventPosition(event, source, position);
    int eventId = ObservationEventId(name);
    if (eventId > 0) {
        RecordObservationEvent(eventId, source, position);
    }
    int soundId = SoundIdForEvent(name);
    if (soundId > 0) {
        RecordSound(soundId, source, position);
    }
    if (StrEqual(name, "smokegrenade_detonate")) {
        RecordSmoke(position);
    }
}

void BuildSoundSlots(int packet[SLBOTS_OBSERVATION_PACKET_WORDS], int base, int observer) {
    int currentTick = GetGameTickCount();
    float observerPosition[3];
    float observerAngles[3];
    GetClientEyePosition(observer, observerPosition);
    GetClientEyeAngles(observer, observerAngles);
    int count = g_SoundHead < SLBOTS_SENSE_CAPACITY ? g_SoundHead : SLBOTS_SENSE_CAPACITY;
    int written = 0;
    for (int age = 0; age < count && written < 16; age++) {
        int index = (g_SoundHead - 1 - age) % SLBOTS_SENSE_CAPACITY;
        int tickAge = currentTick - g_SoundTick[index];
        if (tickAge < 0 || tickAge > 4 * 128) {
            continue;
        }
        float distance = GetVectorDistance(observerPosition, g_SoundPosition[index]);
        if (distance > 1800.0) {
            continue;
        }
        float direction[3];
        float soundAngles[3];
        MakeVectorFromPoints(observerPosition, g_SoundPosition[index], direction);
        GetVectorAngles(direction, soundAngles);
        int category = g_SoundCategory[index] & 0x1F;
        if (SmokeLineOccluded(observerPosition, g_SoundPosition[index])) {
            category |= 1 << 5;
        }
        if (IsClientFlashed(observer)) {
            category |= 1 << 6;
        }
        category |= 1 << 7;
        int bearing = ClampInt(RoundToNearest(NormalizeYaw(soundAngles[1] - observerAngles[1]) / 2.0), -128, 127);
        int distanceBin = ClampInt(RoundToNearest(distance / 256.0), 0, 15);
        int ageBin = ClampInt(RoundToNearest(float(tickAge) / 128.0 / 0.25), 0, 15);
        SetPacketByte(packet, base + 152 + written * 3, category);
        SetPacketByte(packet, base + 153 + written * 3, bearing);
        SetPacketByte(packet, base + 154 + written * 3, (distanceBin << 4) | ageBin);
        written++;
    }
}

void BuildObservationEvents(int packet[SLBOTS_OBSERVATION_PACKET_WORDS], int base, int observer) {
    int currentTick = GetGameTickCount();
    int count = g_EventHead < SLBOTS_SENSE_CAPACITY ? g_EventHead : SLBOTS_SENSE_CAPACITY;
    int written = 0;
    for (int age = 0; age < count && written < 16; age++) {
        int index = (g_EventHead - 1 - age) % SLBOTS_SENSE_CAPACITY;
        int tickAge = currentTick - g_ObservationEventTick[index];
        if (tickAge < 0 || tickAge > 8 * 128) {
            continue;
        }
        int ageSeconds = ClampInt(RoundToNearest(float(tickAge) / 128.0), 0, 7);
        int category = g_ObservationEventCategory[index] & 0x1F;
        if (g_ObservationEventSource[index] == observer) {
            if (category == 5) {
                category = 25;
            } else if (category == 13) {
                category = 26;
            } else if (category == 15) {
                category = 27;
            } else if (category == 18) {
                category = 28;
            } else if (category == 16) {
                category = 29;
            }
        }
        SetPacketByte(packet, base + 200 + written, category | (ageSeconds << 5));
        written++;
    }
}

void BuildCollisionRays(int packet[SLBOTS_OBSERVATION_PACKET_WORDS], int base, int client) {
    float eyePosition[3];
    float eyeAngles[3];
    GetClientEyePosition(client, eyePosition);
    GetClientEyeAngles(client, eyeAngles);
    for (int index = 0; index < 32; index++) {
        float rayAngles[3];
        rayAngles[0] = 0.0;
        rayAngles[1] = eyeAngles[1] + float(index) * 11.25;
        rayAngles[2] = 0.0;
        float direction[3];
        float right[3];
        float up[3];
        GetAngleVectors(rayAngles, direction, right, up);
        float end[3];
        for (int axis = 0; axis < 3; axis++) {
            end[axis] = eyePosition[axis] + direction[axis] * 4096.0;
        }
        Handle trace = TR_TraceRayFilterEx(eyePosition, end, MASK_SOLID, RayType_EndPoint, TraceFilterIgnorePlayers);
        float fraction = trace == null ? 1.0 : TR_GetFraction(trace);
        if (trace != null) {
            delete trace;
        }
        SetPacketByte(packet, base + 216 + index, ClampInt(RoundToNearest(ClampFloat(fraction, 0.0, 1.0) * 63.0), 0, 63));
    }
    for (int index = 0; index < 8; index++) {
        float rayAngles[3];
        rayAngles[0] = -75.0 + float(index) * 21.0;
        rayAngles[1] = eyeAngles[1];
        rayAngles[2] = 0.0;
        float direction[3];
        float right[3];
        float up[3];
        GetAngleVectors(rayAngles, direction, right, up);
        float end[3];
        for (int axis = 0; axis < 3; axis++) {
            end[axis] = eyePosition[axis] + direction[axis] * 4096.0;
        }
        Handle trace = TR_TraceRayFilterEx(eyePosition, end, MASK_SOLID, RayType_EndPoint, TraceFilterIgnorePlayers);
        float fraction = trace == null ? 1.0 : TR_GetFraction(trace);
        if (trace != null) {
            delete trace;
        }
        SetPacketByte(packet, base + 248 + index, ClampInt(RoundToNearest(ClampFloat(fraction, 0.0, 1.0) * 63.0), 0, 63));
    }
}

int CollectBots() {
    int count;
    for (int client = 1; client <= MaxClients && count < SLBOTS_MAX_BOTS; client++) {
        if (!IsClientInGame(client) || !IsFakeClient(client)) {
            continue;
        }
        int team = GetClientTeam(client);
        if (team != CS_TEAM_T && team != CS_TEAM_CT) {
            continue;
        }
        g_BotClients[count++] = client;
    }
    SynchronizeBotSlots(count);
    return count;
}

void SynchronizeBotSlots(int count) {
    for (int slot = 0; slot < SLBOTS_MAX_BOTS; slot++) {
        int client = slot < count ? g_BotClients[slot] : 0;
        int userId = client > 0 ? GetClientUserId(client) : 0;
        if (client == g_BotStateClients[slot] && userId == g_BotStateUserIds[slot]) {
            continue;
        }
        g_BotStateClients[slot] = client;
        g_BotStateUserIds[slot] = userId;
        bool preserveFallback = client > 0 && userId > 0 &&
            g_ClientStateUserIds[client] == userId && g_ClientPermanentFallback[client];
        g_BotPermanentFallback[slot] = preserveFallback;
        if (client > 0 && userId > 0 && g_ClientStateUserIds[client] != userId) {
            g_ClientStateUserIds[client] = userId;
            g_ClientPermanentFallback[client] = false;
        }
        g_MissingConsecutive[slot] = 0;
        g_MissingWindowIndex[slot] = 0;
        g_MissingWindowCount[slot] = 0;
        g_MissingWindowEntries[slot] = 0;
        g_LastMissingTick[slot] = -1;
        g_LastBuyAction[slot] = 0;
        for (int index = 0; index < SLBOTS_MISSING_WINDOW_SIZE; index++) {
            g_MissingWindow[slot][index] = false;
        }
    }
}

int FindBotSlot(int client) {
    for (int slot = 0; slot < g_BotCount; slot++) {
        if (g_BotClients[slot] == client) {
            return slot;
        }
    }
    return -1;
}

int FindActionSlot(int client) {
    for (int slot = 0; slot < SLBOTS_MAX_BOTS; slot++) {
        if (g_LastActionClients[slot] == client) {
            return slot;
        }
    }
    return -1;
}

void RecordMissing(int slot, int tick) {
    if (slot < 0 || slot >= SLBOTS_MAX_BOTS || g_LastMissingTick[slot] == tick) {
        return;
    }
    g_LastMissingTick[slot] = tick;
    g_MissingConsecutive[slot]++;
    int index = g_MissingWindowIndex[slot];
    bool full = g_MissingWindowEntries[slot] >= SLBOTS_MISSING_WINDOW_SIZE;
    if (full && g_MissingWindow[slot][index]) {
        g_MissingWindowCount[slot]--;
    }
    if (!full) {
        g_MissingWindowEntries[slot]++;
    }
    g_MissingWindow[slot][index] = true;
    g_MissingWindowCount[slot]++;
    g_MissingWindowIndex[slot] = (index + 1) % SLBOTS_MISSING_WINDOW_SIZE;
    if (g_MissingConsecutive[slot] >= 32 || g_MissingWindowCount[slot] >= 32) {
        SetPermanentFallback(slot);
    }
}

void RecordValid(int slot, int tick) {
    if (slot < 0 || slot >= SLBOTS_MAX_BOTS || g_LastMissingTick[slot] == tick) {
        return;
    }
    g_LastMissingTick[slot] = tick;
    g_MissingConsecutive[slot] = 0;
    int index = g_MissingWindowIndex[slot];
    bool full = g_MissingWindowEntries[slot] >= SLBOTS_MISSING_WINDOW_SIZE;
    if (full && g_MissingWindow[slot][index]) {
        g_MissingWindowCount[slot]--;
    }
    if (!full) {
        g_MissingWindowEntries[slot]++;
    }
    g_MissingWindow[slot][index] = false;
    g_MissingWindowIndex[slot] = (index + 1) % SLBOTS_MISSING_WINDOW_SIZE;
}

void SetPermanentFallback(int slot) {
    if (slot < 0 || slot >= SLBOTS_MAX_BOTS) {
        return;
    }
    g_BotPermanentFallback[slot] = true;
    int client = g_BotClients[slot];
    if (client > 0 && client <= MaxClients && IsClientInGame(client)) {
        g_ClientStateUserIds[client] = GetClientUserId(client);
        g_ClientPermanentFallback[client] = true;
    }
}

void RecordObservationSnapshot(int serverTick) {
    int index = g_ObservationSnapshotHead % SLBOTS_SENSE_CAPACITY;
    g_ObservationSnapshotTick[index] = serverTick;
    g_ObservationSnapshotBotCount[index] = g_BotCount;
    for (int slot = 0; slot < SLBOTS_MAX_BOTS; slot++) {
        int client = slot < g_BotCount ? g_BotClients[slot] : 0;
        g_ObservationSnapshotClients[index][slot] = client;
        g_ObservationSnapshotUserIds[index][slot] = client > 0 ? GetClientUserId(client) : 0;
    }
    g_ObservationSnapshotHead++;
    if (g_ObservationSnapshotCount < SLBOTS_SENSE_CAPACITY) {
        g_ObservationSnapshotCount++;
    }
}

int FindObservationSnapshot(int serverTick) {
    for (int age = 0; age < g_ObservationSnapshotCount; age++) {
        int index = (g_ObservationSnapshotHead - 1 - age) % SLBOTS_SENSE_CAPACITY;
        if (g_ObservationSnapshotTick[index] == serverTick) {
            return index;
        }
    }
    return -1;
}

void PollActionPacket() {
    int packet[SLBOTS_ACTION_PACKET_WORDS];
    int result = SLBots_TryReadAction(packet);
    if (result <= 0) {
        return;
    }
    if (GetPacketInt32(packet, 12) != g_Epoch ||
        GetPacketInt16(packet, 8) != SLBOTS_SCHEMA_VERSION ||
        GetPacketInt16(packet, 10) != SLBOTS_ACTION_PACKET_BYTES ||
        GetPacketInt16(packet, 22) != 0) {
        return;
    }
    int packetTick = GetPacketInt32(packet, 16);
    int packetBotCount = GetPacketInt16(packet, 20);
    if (packetBotCount < 0 || packetBotCount > SLBOTS_MAX_BOTS) {
        return;
    }
    if (packetTick < g_LastActionTick) {
        return;
    }
    int snapshot = FindObservationSnapshot(packetTick);
    if (snapshot < 0 || g_ObservationSnapshotBotCount[snapshot] != packetBotCount) {
        return;
    }
    for (int index = 0; index < packetBotCount; index++) {
        int client = g_ObservationSnapshotClients[snapshot][index];
        if (client <= 0 || !IsClientInGame(client) ||
            GetClientUserId(client) != g_ObservationSnapshotUserIds[snapshot][index]) {
            return;
        }
    }
    for (int index = 0; index < SLBOTS_ACTION_PACKET_WORDS; index++) {
        g_LastActionPacket[index] = packet[index];
    }
    for (int index = 0; index < SLBOTS_MAX_BOTS; index++) {
        int client = index < packetBotCount ? g_ObservationSnapshotClients[snapshot][index] : 0;
        g_LastActionClients[index] = client;
    }
    g_LastActionTick = packetTick + 1;
    g_LastActionAckTick = -1;
    g_LastActionPacketReceivedTime = GetEngineTime();
}

void RecordActionApplied(int serverTick) {
    if (g_LastActionAckTick == serverTick || g_LastActionPacketReceivedTime <= 0.0) {
        return;
    }
    int latencyUs = RoundToNearest((GetEngineTime() - g_LastActionPacketReceivedTime) * 1000000.0);
    if (latencyUs < 0) {
        latencyUs = 0;
    }
    g_LastActionAckTick = serverTick;
    PublishControlEvent(SLBOTS_CONTROL_LATENCY_SAMPLE, serverTick, g_RoundNumber, latencyUs);
}

void ApplyNeutralAction(
    int client,
    int &buttons,
    int &impulse,
    float vel[3],
    float angles[3],
    int &weapon,
    int &subtype,
    int mouse[2]
) {
    buttons = 0;
    impulse = 0;
    vel[0] = 0.0;
    vel[1] = 0.0;
    vel[2] = 0.0;
    float currentAngles[3];
    GetClientEyeAngles(client, currentAngles);
    angles[0] = currentAngles[0];
    angles[1] = currentAngles[1];
    angles[2] = currentAngles[2];
    int currentWeapon = CurrentWeaponEntity(client);
    if (currentWeapon > 0) {
        weapon = currentWeapon;
    }
    subtype = 0;
    mouse[0] = 0;
    mouse[1] = 0;
}

void ExecuteBuyAction(int client, int slot, int action) {
    if (action <= 0 || action == g_LastBuyAction[slot]) {
        return;
    }
    char item[32];
    switch (action) {
        case 1: strcopy(item, sizeof(item), "vest");
        case 2: strcopy(item, sizeof(item), "vesthelm");
        case 3: strcopy(item, sizeof(item), "ak47");
        case 4: strcopy(item, sizeof(item), "m4a1");
        case 5: strcopy(item, sizeof(item), "smokegrenade");
        case 6: strcopy(item, sizeof(item), "flashbang");
        case 7: strcopy(item, sizeof(item), "hegrenade");
        case 8: strcopy(item, sizeof(item), "molotov");
        case 9: strcopy(item, sizeof(item), "defuser");
        default: return;
    }
    FakeClientCommand(client, "buy %s", item);
    g_LastBuyAction[slot] = action;
}

float GetActionFloat(int wordOffset) {
    return view_as<float>(g_LastActionPacket[wordOffset]);
}

int Signed16(int value) {
    int result = value & 0xFFFF;
    return result >= 0x8000 ? result - 0x10000 : result;
}

float NormalizeYaw(float value) {
    while (value > 180.0) {
        value -= 360.0;
    }
    while (value < -180.0) {
        value += 360.0;
    }
    return value;
}

float NormalizePitch(float value) {
    return ClampFloat(value, -89.0, 89.0);
}

int ClampInt(int value, int minimum, int maximum) {
    if (value < minimum) {
        return minimum;
    }
    if (value > maximum) {
        return maximum;
    }
    return value;
}

float ClampFloat(float value, float minimum, float maximum) {
    if (value < minimum) {
        return minimum;
    }
    if (value > maximum) {
        return maximum;
    }
    return value;
}

void PublishControlEvent(int eventType, int serverTick, int roundNumber, int valueUs = 0) {
    if (!g_Open) {
        return;
    }
    int packet[SLBOTS_CONTROL_EVENT_WORDS];
    for (int index = 0; index < SLBOTS_CONTROL_EVENT_WORDS; index++) {
        packet[index] = 0;
    }
    SetPacketByte(packet, 0, 'S');
    SetPacketByte(packet, 1, 'L');
    SetPacketByte(packet, 2, 'B');
    SetPacketByte(packet, 3, 'C');
    SetPacketByte(packet, 4, 'T');
    SetPacketByte(packet, 5, 'L');
    SetPacketByte(packet, 6, '1');
    SetPacketInt16(packet, 8, SLBOTS_CONTROL_SCHEMA_VERSION);
    SetPacketInt16(packet, 10, eventType);
    g_ControlSequence++;
    SetPacketInt64(packet, 12, g_ControlSequence, 0);
    SetPacketInt32(packet, 20, g_Epoch);
    SetPacketInt32(packet, 24, serverTick);
    SetPacketInt16(packet, 28, g_Get5Available ? Get5_GetGameState() : -1);
    SetPacketInt16(packet, 30, g_Get5MapNumberAvailable ? Get5_GetMapNumber() : SLBOTS_MAP_ID_MIRAGE);
    SetPacketInt16(packet, 32, roundNumber);
    SetPacketInt16(packet, 34, GetTeamScore(CS_TEAM_CT));
    SetPacketInt16(packet, 36, GetTeamScore(CS_TEAM_T));
    SetPacketInt32(packet, 38, g_ControlPolicyGeneration);
    SetPacketInt64(packet, 42, valueUs, 0);
    int matchHashLow = 0;
    int matchHashHigh = 0;
    if (g_Get5MatchIdAvailable) {
        char matchId[128];
        Get5_GetMatchID(matchId, sizeof(matchId));
        matchHashLow = HashText32(matchId, 0x811C9DC5);
        matchHashHigh = HashText32(matchId, 0x01000193);
    }
    SetPacketInt64(packet, 50, matchHashLow, matchHashHigh);
    SetPacketInt32(packet, 58, ControlCrc32(packet, SLBOTS_CONTROL_EVENT_BYTES));
    if (SLBots_PublishControl(packet) <= 0) {
        LogError("SL-Bots control event publish failed: type=%d sequence=%d", eventType, g_ControlSequence);
    }
}

int HashText32(const char[] value, int seed) {
    int hash = seed;
    for (int index = 0; value[index] != '\0'; index++) {
        hash ^= value[index];
        hash *= 0x01000193;
    }
    return hash;
}

void SetPacketInt64(int[] packet, int byteOffset, int low, int high) {
    SetPacketInt32(packet, byteOffset, low);
    SetPacketInt32(packet, byteOffset + 4, high);
}

int GetPacketByte(const int[] packet, int byteOffset) {
    return (packet[byteOffset / 4] >> ((byteOffset % 4) * 8)) & 0xFF;
}

void SetPacketByte(int[] packet, int byteOffset, int value) {
    int wordOffset = byteOffset / 4;
    int shift = (byteOffset % 4) * 8;
    packet[wordOffset] = (packet[wordOffset] & ~(0xFF << shift)) | ((value & 0xFF) << shift);
}

int GetPacketInt16(const int[] packet, int byteOffset) {
    return GetPacketByte(packet, byteOffset) | (GetPacketByte(packet, byteOffset + 1) << 8);
}

int GetPacketInt32(const int[] packet, int byteOffset) {
    return GetPacketByte(packet, byteOffset) |
        (GetPacketByte(packet, byteOffset + 1) << 8) |
        (GetPacketByte(packet, byteOffset + 2) << 16) |
        (GetPacketByte(packet, byteOffset + 3) << 24);
}

void SetPacketInt16(int[] packet, int byteOffset, int value) {
    SetPacketByte(packet, byteOffset, value);
    SetPacketByte(packet, byteOffset + 1, value >> 8);
}

void SetPacketInt32(int[] packet, int byteOffset, int value) {
    SetPacketByte(packet, byteOffset, value);
    SetPacketByte(packet, byteOffset + 1, value >> 8);
    SetPacketByte(packet, byteOffset + 2, value >> 16);
    SetPacketByte(packet, byteOffset + 3, value >> 24);
}

int Crc32(const int[] packet, int byteLength) {
    int crc = -1;
    for (int index = 0; index < byteLength; index++) {
        int value = GetPacketByte(packet, index);
        if (index >= 32 && index < 36) {
            value = 0;
        }
        crc ^= value;
        for (int bit = 0; bit < 8; bit++) {
            int mask = -(crc & 1);
            crc = (crc >>> 1) ^ (0xEDB88320 & mask);
        }
    }
    return crc ^ -1;
}

int ControlCrc32(const int[] packet, int byteLength) {
    int crc = -1;
    for (int index = 0; index < byteLength; index++) {
        int value = GetPacketByte(packet, index);
        if (index >= 58 && index < 62) {
            value = 0;
        }
        crc ^= value;
        for (int bit = 0; bit < 8; bit++) {
            int mask = -(crc & 1);
            crc = (crc >>> 1) ^ (0xEDB88320 & mask);
        }
    }
    return crc ^ -1;
}
