/*
 * Load server configuration. Most configuration are loaded
 * from the environment variables.
 */
'use strict';
var path = require('path');
var env = require('./environment.js');
var constant = require('./constant.js');

exports.load = function() {
    // Environment variables.
    var envx = env.load();

    // Server log.
    var serLogLevel = envx[constant.ENV.SERVER_LOG_LEVEL] ?
        envx[constant.ENV.SERVER_LOG_LEVEL] : 'error';
    var serLogPath = envx[constant.ENV.SERVER_LOG_PATH] ?
        envx[constant.ENV.SERVER_LOG_PATH] : path.join(process.cwd(), '/var/log/');
    var serLogFile = path.join(serLogPath, 'cagent_nodejs.log');

    // Function log.
    var funcLogPath = envx[constant.ENV.FUNC_LOG_PATH] ?
        envx[constant.ENV.FUNC_LOG_PATH] : path.join(process.cwd(), '/var/log/');
    var funcLogFile = path.join(funcLogPath, 'func_nodejs.log');

    return {
        // Environment variables.
        env: envx,

        // Server configurations.
        server: {
            // port
            port: envx[constant.ENV.SERVER_PORT],

            // Log
            logLevel: serLogLevel,
            logFile: serLogFile,
        },

        // Function configurations.
        func: {
            // Function code.
            codePath: envx[constant.ENV.FUNC_CODE_PATH],

            // Log.
            logLevel: 'silly',
            logFile: funcLogFile,

        },

        // Global temporary data store.
        cache: {}
    };
};
