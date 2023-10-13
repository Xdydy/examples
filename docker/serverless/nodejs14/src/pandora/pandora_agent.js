const defaultConfig = require('./pandora_config');

async function start() {
  const { CoreSDK } = require('@pandorajs/core-sdk');

  const sdk = new CoreSDK({
    mode: 'supervisor',
    appName: 'aliyun-fc-nodejs',
    processName: 'alinode-remote-debug-service',
    extendConfig: [
      {
        config: defaultConfig,
        configDir: process.env.FC_FUNC_CODE_PATH,
      },
    ]
  });

  sdk.instantiate();
  console.log(`[Pandora.js] Pandora.js remote debug service instantiated!`);

  try {
    await sdk.start();
  } catch (error) {
    console.error('[Pandora.js] Start Pandora.js remote debug service went wrong!', error);
    return process.exit(1);
  }

  console.log(`[Pandora.js] Pandora.js remote debug service started!`);
};

start().catch((error) => {
  console.error('[Pandora.js] Start Pandora.js remote debug service went wrong!', error);
  return process.exit(1);
});