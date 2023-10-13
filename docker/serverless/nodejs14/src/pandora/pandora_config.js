module.exports = {
  components: {
    actuatorServer: {
      path: require.resolve('@pandorajs/component-actuator-server'),
    },
    processInfo: {
      path: require.resolve('@pandorajs/component-process-info'),
    },
    remoteDebug: {
      path: require.resolve('./component-remote-debug/ComponentRemoteDebug'),
    },
  },
  remoteDebug: {
    port: 8000,
  },
  actuatorServer: {
    http: {
      host: '0.0.0.0',
      port: 12580
    },
  },
};