#ifndef SL_BOTS_SOURCEMOD_CONFIG_H
#define SL_BOTS_SOURCEMOD_CONFIG_H

#define SMEXT_CONF_NAME "SL-Bots IPC"
#define SMEXT_CONF_DESCRIPTION "SL-Bots shared-memory IPC extension"
#define SMEXT_CONF_VERSION "0.1.0"
#define SMEXT_CONF_AUTHOR "SL-Bots"
#define SMEXT_CONF_URL ""
#define SMEXT_CONF_LOGTAG "SLBOTS"
#define SMEXT_CONF_LICENSE "MIT"
#define SMEXT_CONF_DATESTRING __DATE__

#define SMEXT_LINK(name) SDKExtension *g_pExtensionIface = name;

#endif
