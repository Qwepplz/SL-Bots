#include <cstdint>

#include "smsdk_ext.h"

extern "C" {
int SLBots_Open(const char* name, std::uint32_t epoch, std::uint32_t capacity) noexcept;
int SLBots_PublishObservation(const void* packet) noexcept;
int SLBots_TryReadAction(void* packet) noexcept;
std::uint32_t SLBots_GetHeartbeat() noexcept;
int SLBots_SwitchEpoch(std::uint32_t epoch) noexcept;
void SLBots_Abort() noexcept;
void SLBots_Close() noexcept;
}

namespace {

cell_t NativeSLBotsOpen(SourcePawn::IPluginContext* context, const cell_t* params) {
    char* name = nullptr;
    if (context->LocalToString(params[1], &name) != SP_ERROR_NONE) {
        return context->ThrowNativeError("name is not a valid string");
    }
    return SLBots_Open(
        name,
        static_cast<std::uint32_t>(params[2]),
        static_cast<std::uint32_t>(params[3]));
}

cell_t NativeSLBotsPublishObservation(
    SourcePawn::IPluginContext* context,
    const cell_t* params) {
    cell_t* packet = nullptr;
    if (context->LocalToPhysAddr(params[1], &packet) != SP_ERROR_NONE) {
        return context->ThrowNativeError("packet is not a valid array");
    }
    return SLBots_PublishObservation(packet);
}

cell_t NativeSLBotsTryReadAction(
    SourcePawn::IPluginContext* context,
    const cell_t* params) {
    cell_t* packet = nullptr;
    if (context->LocalToPhysAddr(params[1], &packet) != SP_ERROR_NONE) {
        return context->ThrowNativeError("packet is not a valid array");
    }
    return SLBots_TryReadAction(packet);
}

cell_t NativeSLBotsGetHeartbeat(SourcePawn::IPluginContext*, const cell_t*) {
    return static_cast<cell_t>(SLBots_GetHeartbeat());
}

cell_t NativeSLBotsSwitchEpoch(SourcePawn::IPluginContext*, const cell_t* params) {
    return SLBots_SwitchEpoch(static_cast<std::uint32_t>(params[1]));
}

cell_t NativeSLBotsAbort(SourcePawn::IPluginContext*, const cell_t*) {
    SLBots_Abort();
    return 0;
}

cell_t NativeSLBotsClose(SourcePawn::IPluginContext*, const cell_t*) {
    SLBots_Close();
    return 0;
}

const sp_nativeinfo_t g_SLBotsNatives[] = {
    {"SLBots_Open", NativeSLBotsOpen},
    {"SLBots_PublishObservation", NativeSLBotsPublishObservation},
    {"SLBots_TryReadAction", NativeSLBotsTryReadAction},
    {"SLBots_GetHeartbeat", NativeSLBotsGetHeartbeat},
    {"SLBots_SwitchEpoch", NativeSLBotsSwitchEpoch},
    {"SLBots_Abort", NativeSLBotsAbort},
    {"SLBots_Close", NativeSLBotsClose},
    {nullptr, nullptr},
};

class SLBotsExtension final : public SDKExtension {
public:
    bool SDK_OnLoad(char*, size_t, bool) override {
        g_pShareSys->AddNatives(myself, g_SLBotsNatives);
        return true;
    }

    void SDK_OnUnload() override {
        SLBots_Close();
    }
};

}

SLBotsExtension g_SLBotsExtension;

SMEXT_LINK(&g_SLBotsExtension);
