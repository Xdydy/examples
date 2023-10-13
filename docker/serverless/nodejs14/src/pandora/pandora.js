exports.start = async function() {
  const defaultConfig = require('./pandora_config');
  const { CoreSDK } = require('@pandorajs/core-sdk');

  const sdk = new CoreSDK({
    mode: 'worker',
    appName: 'aliyun-fc-nodejs',
    processName: 'alinode-remote-debug-client',
    extendConfig: [
      {
        config: defaultConfig,
        configDir: process.env.FC_FUNC_CODE_PATH,
      },
    ]
  });

  sdk.instantiate();

  await sdk.start();
};