#pragma semicolon 1
#pragma newdecls required
#include <sourcemod>
#include <sdktools>
native int Get5_GetGameState();
bool g_Active;
int g_Round;
int g_FreezeEvents, g_FreezeState=-2;
int LiveState() {
    ConVar state=FindConVar("get5_game_state");
    if (state!=null) { return state.IntValue; }
    return GetFeatureStatus(FeatureType_Native,"Get5_GetGameState")==FeatureStatus_Available ? Get5_GetGameState() : -1;
}
int g_Shots[MAXPLAYERS + 1], g_Damage[MAXPLAYERS + 1], g_Friendly[MAXPLAYERS + 1];
int g_Kills[MAXPLAYERS + 1], g_Deaths[MAXPLAYERS + 1];
int g_UserId[MAXPLAYERS + 1];
void ResetClientCounters(int client) {
    g_UserId[client]=0;
    g_Shots[client]=0; g_Damage[client]=0; g_Friendly[client]=0;
    g_Kills[client]=0; g_Deaths[client]=0;
}
void EnsureClientIdentity(int client) {
    int userid=GetClientUserId(client);
    if (g_UserId[client]!=userid) {
        ResetClientCounters(client);
        g_UserId[client]=userid;
    }
}
public void OnClientPutInServer(int client) { ResetClientCounters(client); }
public void OnClientDisconnect(int client) { ResetClientCounters(client); }
File g_Teacher;
public Plugin myinfo = {name="SL-Bots temporary evaluation", author="SL-Bots", description="Read-only combat counters and teacher UserCmd capture", version="1.0"};
public void OnPluginStart() {
    MarkNativeAsOptional("Get5_GetGameState");
    HookEvent("round_start", RoundStart);
    HookEvent("round_freeze_end", FreezeEnd);
    HookEvent("round_end", RoundEnd);
    HookEvent("weapon_fire", Fire);
    HookEvent("player_hurt", Hurt);
    HookEvent("player_death", Death);
    RegServerCmd("sl_bots_eval_status", Status);
    RegServerCmd("sl_bots_eval_angles", AngleProbe);
    RegServerCmd("sl_bots_teacher_start", StartTeacher);
    RegServerCmd("sl_bots_teacher_stop", StopTeacher);
}
public void OnMapStart() {
    g_Active=false; g_Round=0;
    for (int i=1;i<=MaxClients;i++) { ResetClientCounters(i); }
}
public void OnPluginEnd() { delete g_Teacher; }
public void RoundStart(Event event,const char[] name,bool broadcast) { g_Active=false; g_Round++; }
public void RoundEnd(Event event,const char[] name,bool broadcast) { g_Active=false; }
public void FreezeEnd(Event event,const char[] name,bool broadcast) {
    g_FreezeEvents++; g_FreezeState=LiveState();
    g_Active=LiveState()==7;
}
bool Bot(int client) {
    if (client<=0 || client>MaxClients || !IsClientInGame(client) || !IsFakeClient(client) || IsClientSourceTV(client)) { return false; }
    // A slot is reusable; only its current connection owns these counters.
    EnsureClientIdentity(client);
    return true;
}
public void Fire(Event event,const char[] name,bool broadcast) {
    int client=GetClientOfUserId(event.GetInt("userid"));
    if (g_Active && Bot(client)) { g_Shots[client]++; }
}
public void Hurt(Event event,const char[] name,bool broadcast) {
    int attacker=GetClientOfUserId(event.GetInt("attacker")), victim=GetClientOfUserId(event.GetInt("userid"));
    if (!g_Active || !Bot(attacker) || !Bot(victim) || attacker==victim) { return; }
    if (GetClientTeam(attacker)==GetClientTeam(victim)) { g_Friendly[attacker]+=event.GetInt("dmg_health"); }
    else { g_Damage[attacker]+=event.GetInt("dmg_health"); }
}
public void Death(Event event,const char[] name,bool broadcast) {
    int attacker=GetClientOfUserId(event.GetInt("attacker")), victim=GetClientOfUserId(event.GetInt("userid"));
    if (!g_Active || !Bot(victim)) { return; }
    g_Deaths[victim]++;
    if (Bot(attacker) && attacker!=victim && GetClientTeam(attacker)!=GetClientTeam(victim)) { g_Kills[attacker]++; }
}
public Action Status(int args) {
    PrintToServer("{\"tick\":%d,\"round\":%d,\"active\":%d,\"state\":%d,\"native_available\":%d,\"freeze\":%d,\"freeze_events\":%d,\"freeze_state\":%d}",GetGameTickCount(),g_Round,g_Active,LiveState(),GetFeatureStatus(FeatureType_Native,"Get5_GetGameState")==FeatureStatus_Available,GameRules_GetProp("m_bFreezePeriod"),g_FreezeEvents,g_FreezeState);
    for(int i=1;i<=MaxClients;i++) {
        if (!IsClientInGame(i) || !IsFakeClient(i)) { continue; }
        EnsureClientIdentity(i);
        PrintToServer("{\"userid\":%d,\"team\":%d,\"alive\":%d,\"tv\":%d,\"shots\":%d,\"enemy_damage\":%d,\"friendly_damage\":%d,\"kills\":%d,\"deaths\":%d}",GetClientUserId(i),GetClientTeam(i),IsPlayerAlive(i),IsClientSourceTV(i),g_Shots[i],g_Damage[i],g_Friendly[i],g_Kills[i],g_Deaths[i]);
    }
    return Plugin_Handled;
}
public Action StartTeacher(int args) {
    if (args!=1 || g_Teacher!=null) { PrintToServer("teacher start rejected"); return Plugin_Handled; }
    char name[128], path[256]; GetCmdArg(1,name,sizeof(name));
    for(int i=0;name[i]!='\0';i++) {
        if (!((name[i]>='a' && name[i]<='z') || (name[i]>='0' && name[i]<='9') || name[i]=='-' || name[i]=='_')) { PrintToServer("invalid teacher name"); return Plugin_Handled; }
    }
    Format(path,sizeof(path),"addons/sourcemod/logs/%s.csv",name);
    if (FileExists(path)) { PrintToServer("teacher file already exists"); return Plugin_Handled; }
    g_Teacher=OpenFile(path,"w");
    if (g_Teacher!=null) { g_Teacher.WriteLine("tick,round,userid,forward,side,up,yaw,pitch,view_yaw,view_pitch,buttons"); PrintToServer("teacher capture started"); }
    return Plugin_Handled;
}
public Action StopTeacher(int args) { delete g_Teacher; PrintToServer("teacher capture stopped"); return Plugin_Handled; }
public Action OnPlayerRunCmd(int client,int &buttons,int &impulse,float vel[3],float angles[3],int &weapon,int &subtype,int &cmdnum,int &tickcount,int &seed,int mouse[2]) {
    if (!g_Active || g_Teacher==null || !Bot(client) || !IsPlayerAlive(client)) { return Plugin_Continue; }
    float view[3]; GetClientEyeAngles(client,view);
    g_Teacher.WriteLine("%d,%d,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%d",GetGameTickCount(),g_Round,GetClientUserId(client),vel[0],vel[1],vel[2],angles[1],angles[0],view[1],view[0],buttons);
    return Plugin_Continue;
}

public Action AngleProbe(int args) {
    float vectors[3][3]={{100.0,0.0,10.0},{100.0,0.0,-10.0},{100.0,0.0,0.0}};
    for(int i=0;i<3;i++) {
        float angles[3]; GetVectorAngles(vectors[i],angles);
        PrintToServer("{\"case\":%d,\"z\":%.2f,\"pitch\":%.5f,\"encoded_pitch\":%d}",i,vectors[i][2],angles[0],RoundToNearest(angles[0])>127?127:RoundToNearest(angles[0]));
    }
    return Plugin_Handled;
}
